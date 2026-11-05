"""Flat-ground encoder/IMU dead reckoning; no world pose or velocity input.

Q1-specific FK and equal-radius rolling wheels. This is a nominal rolling
observer, not a contact/slip estimator. Startup roll/pitch may be supplied by
an IMU calibration; the default starts from the first gravity measurement.
"""
import torch


def wrap(angle):
    return torch.atan2(torch.sin(angle),torch.cos(angle))


class WheelOdometry:
    def __init__(self,count,device,radius=.07496,l1=.2,l2=.2,gravity_gain=.02):
        self.radius,self.l1,self.l2=radius,l1,l2
        self.gain=gravity_gain
        self.xy=torch.zeros(count,2,device=device)
        self.yaw=torch.zeros(count,device=device)
        self.rp=torch.zeros(count,2,device=device)
        self.previous_q=torch.zeros(count,6,device=device)
        self.previous_center=torch.zeros(count,2,2,device=device)
        self.initialized=torch.zeros(count,dtype=torch.bool,device=device)
        self.count=torch.zeros(count,dtype=torch.long,device=device)
        self.initial_q=torch.zeros_like(self.previous_q)
        self.initial_rp=torch.zeros_like(self.rp)
        self.integrated_rp=torch.zeros_like(self.rp)
        self.encoder_y=torch.tensor([.14927,-.14973],device=device)

    def angles(self,gravity):
        return torch.stack((torch.atan2(-gravity[:,1],-gravity[:,2]),
            torch.atan2(gravity[:,0],gravity[:,1:].norm(dim=-1))),dim=-1)

    def centers(self,q,rp):
        hip=q[:,[0,3]];calf=q[:,[1,4]]
        x=-.025225-self.l1*hip.sin()-self.l2*(hip+calf).sin()
        z=.0084361-self.l1*hip.cos()-self.l2*(hip+calf).cos()
        roll,pitch=rp[:,0:1],rp[:,1:2]
        y=self.encoder_y[None].expand_as(x)
        rotated_y=roll.cos()*y-roll.sin()*z
        rotated_z=roll.sin()*y+roll.cos()*z
        rotated_x=pitch.cos()*x+pitch.sin()*rotated_z
        return torch.stack((rotated_x,rotated_y),dim=-1)

    def reset(self,ids,q,gravity,calibrated_rp=None):
        self.xy[ids]=0;self.yaw[ids]=0
        self.rp[ids]=self.angles(gravity)[ids] if calibrated_rp is None else calibrated_rp[ids]
        self.previous_q[ids]=q[ids]
        self.previous_center[ids]=self.centers(q,self.rp)[ids]
        self.initial_q[ids]=q[ids];self.initial_rp[ids]=self.rp[ids]
        self.integrated_rp[ids]=0;self.count[ids]=0
        self.initialized[ids]=True

    def update(self,q,gyro,gravity,dt):
        ids=(~self.initialized).nonzero(as_tuple=False).flatten()
        if len(ids):self.reset(ids,q,gravity)
        old_pitch=self.rp[:,1].clone()
        roll,pitch=self.rp[:,0],self.rp[:,1]
        rates=torch.stack((gyro[:,0]+roll.sin()*pitch.tan()*gyro[:,1]+roll.cos()*pitch.tan()*gyro[:,2],
                           roll.cos()*gyro[:,1]-roll.sin()*gyro[:,2]),dim=-1)
        self.integrated_rp+=rates*dt
        self.rp+=rates*dt
        gain=torch.where(self.count<30,.1,self.gain)[:,None]
        self.rp+=gain*wrap(self.angles(gravity)-self.rp)
        center=self.centers(q,self.rp)
        delta_q=wrap(q-self.previous_q)
        wheel_delta=delta_q[:,[2,5]]+delta_q[:,[0,3]]+delta_q[:,[1,4]]
        wheel_delta+=wrap(self.rp[:,1]-old_pitch)[:,None]
        displacement=self.radius*wheel_delta
        relative_delta=center-self.previous_center
        forward=displacement-relative_delta[:,:,0]
        track=(center[:,0,1]-center[:,1,1]).clamp_min(.1)
        delta_yaw=(forward[:,1]-forward[:,0])/track
        dx=forward.mean(-1)+delta_yaw*center[:,:,1].mean(-1)
        dy=-relative_delta[:,:,1].mean(-1)-delta_yaw*center[:,:,0].mean(-1)
        middle=self.yaw+.5*delta_yaw
        self.xy[:,0]+=middle.cos()*dx-middle.sin()*dy
        self.xy[:,1]+=middle.sin()*dx+middle.cos()*dy
        self.count+=1
        calibrate=self.count==30
        # Correct the origin's initial IMU sample bias using 0.3 s of sensor
        # history and integrated rotation. No simulation pose is involved.
        if calibrate.any():
            refined_initial=self.rp-self.integrated_rp
            old_center=self.centers(self.initial_q,self.initial_rp).mean(1)
            new_center=self.centers(self.initial_q,refined_initial).mean(1)
            correction=new_center-old_center
            correction[:,0]-=self.radius*wrap(refined_initial[:,1]-self.initial_rp[:,1])
            self.xy[calibrate]+=correction[calibrate]
        self.yaw=wrap(self.yaw+delta_yaw)
        self.previous_q[:]=q
        self.previous_center[:]=center
        return self.xy
