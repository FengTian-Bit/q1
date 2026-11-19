"""Tensor-only label geometry; all velocities are in world coordinates."""

import torch


def quat_apply(q, vector):
    cross = torch.cross(q[..., :3], vector, dim=-1)
    return vector + 2*(q[..., 3:4]*cross + torch.cross(q[..., :3], cross, dim=-1))


def flat_wheel_slip(states, center_local, com_local, radius, width, base_quat):
    """Signed bottom-point speed along the body's projected forward direction.

    PhysX rigid-body linear velocity is measured at COM, while the pose uses
    the link frame. Include COM/collision offsets and absolute angular velocity.
    This is a geometric rolling-contact approximation, not a mesh contact query.
    """
    q = states[..., 3:7]
    axis_local = torch.zeros_like(center_local)
    axis_local[..., 1] = 1.0
    axis = quat_apply(q, axis_local)
    normal = torch.zeros_like(axis)
    normal[..., 2] = 1.0
    axial = (normal*axis).sum(-1, keepdim=True)
    radial = normal - axial*axis
    bottom = -radius*radial/radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    bottom -= 0.5*width*axial.sign()*axis
    arm = quat_apply(q, center_local-com_local) + bottom
    point_velocity = states[..., 7:10] + torch.cross(states[..., 10:13], arm, dim=-1)
    forward_local = torch.zeros_like(base_quat[..., :3])
    forward_local[..., 0] = 1.0
    forward = quat_apply(base_quat, forward_local)
    forward[..., 2] = 0.0
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (point_velocity*forward[:, None]).sum(-1)


def update_contact(force, contact, counter, on_force, off_force, confirm_frames):
    """Schmitt trigger with a consecutive-frame confirmation for transitions."""
    candidate = torch.where(contact, force > off_force, force >= on_force)
    counter = torch.where(candidate != contact, counter + 1, torch.zeros_like(counter))
    switch = counter >= confirm_frames
    contact = torch.where(switch, candidate, contact)
    return contact, torch.where(switch, torch.zeros_like(counter), counter)
