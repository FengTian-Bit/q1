"""Guard unsafe PPO steps, preserve Adam transfer, and reject incomplete gates."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np

from test_q1_tcn import SCOPE,TrainConfig,tensor_env
from test_q1_flat import load_code


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_high_kl_is_rejected_before_any_optimizer_step(self):
        cfg=TrainConfig()
        cfg.runner.num_steps_per_env=2
        cfg.algorithm.max_kl=.02
        cfg.algorithm.desired_kl=None
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            runner=SCOPE['TCNPpoRunner'](tensor_env(),cfg,folder)
            batch,last,_,_=runner.collect()
            with torch.no_grad():runner.model.actor[-1].bias.add_(1)
            before=copy.deepcopy(runner.model.state_dict())
            result=runner.alg.update(batch,last)
            self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual(result['kl_stop'],1)
            self.assertGreater(result['kl_max'],.02)
            self.assertFalse(runner.alg.optimizer.state)
            for key,value in before.items():torch.testing.assert_close(runner.model.state_dict()[key],value)

    def test_adam_moments_transfer_keeps_chosen_small_learning_rate(self):
        scope=dict(SCOPE);scope['__name__']='stability_test_helpers'
        load_code('tools/q1_reward_iteration.py',scope)
        cfg=TrainConfig();cfg.runner.num_steps_per_env=2
        cfg.algorithm.num_learning_epochs=1
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            old=SCOPE['TCNPpoRunner'](tensor_env(),cfg,Path(folder)/'old')
            batch,last,_,_=old.collect();old.alg.update(batch,last)
            saved=dict(model_state_dict=old.model.state_dict(),optimizer_state_dict=copy.deepcopy(old.alg.optimizer.state_dict()))
            cfg.algorithm.learning_rate=3e-5
            new=SCOPE['TCNPpoRunner'](tensor_env(),cfg,Path(folder)/'new')
            scope['load_checkpoint']=lambda _:saved
            scope['inherit_optimizer'](new,None)
            self.assertEqual(new.alg.learning_rate,3e-5)
            self.assertTrue(all(group['lr']==3e-5 for group in new.alg.optimizer.param_groups))
            self.assertEqual(new.iteration,0)
            for old_state,new_state in zip(saved['optimizer_state_dict']['state'].values(),new.alg.optimizer.state_dict()['state'].values()):
                for key in ['step','exp_avg','exp_avg_sq']:torch.testing.assert_close(old_state[key],new_state[key])

    def test_gate_rejects_failed_missing_and_biased_results(self):
        metrics={key:dict(mean=0.,n=64) for key in ['failed','steady_speed_rmse_m_s',
            'forward_speed_bias_m_s','reverse_speed_bias_m_s','stop_speed_rmse_m_s',
            'locked_parking_end_m','pitch_rms_rad','roll_rms_rad']}
        metrics['command_settled_fraction']=dict(mean=1.,n=64)
        good=dict(trials=64,metrics=metrics)
        limits=dict(steady_rmse=.03,bias=.025,settled_fraction=.98,stop_rmse=.025,stand_drift=.025,stand_tilt_deg=3.5)
        gate=SCOPE['tracking_gate']
        self.assertTrue(gate(good,good,limits)[0])
        for key,value in [('failed',1/64),('forward_speed_bias_m_s',-.1),('command_settled_fraction',None)]:
            bad=copy.deepcopy(good);bad['metrics'][key]['mean']=value
            self.assertFalse(gate(bad,good,limits)[0])
        bad=copy.deepcopy(good);bad['metrics']['steady_speed_rmse_m_s']['n']=63
        self.assertFalse(gate(bad,good,limits)[0])

    def test_edge_sampling_preserves_tensor_and_both_directions(self):
        scope=dict(SCOPE);scope['__name__']='stability_test_helpers'
        load_code('tools/q1_reward_iteration.py',scope)
        env=tensor_env(100)
        snapshot=dict(env=SCOPE['config_dict'](env.cfg))
        reference=env.command_ranges['lin_vel_x']
        scope['apply_speed_limit'](env,snapshot,.3,.4)
        self.assertIs(reference,env.command_ranges['lin_vel_x'])
        self.assertEqual(int((reference[:,0]>0).sum()),20)
        self.assertEqual(int((reference[:,1]<0).sum()),20)
        self.assertTrue((reference[:,0]<reference[:,1]).all())
        self.assertLessEqual(float(reference.abs().max()),.300001)

    def test_lower_speed_failure_blocks_target_promotion(self):
        scope=dict(SCOPE);scope['__name__']='stability_test_helpers'
        load_code('tools/q1_reward_iteration.py',scope)
        commands=SimpleNamespace(ranges=SimpleNamespace(lin_vel_x=[-.6,.6]))
        runner=SimpleNamespace(iteration=0,snapshot={},
            cfg=SimpleNamespace(runner=SimpleNamespace(max_iterations=100)),
            env=SimpleNamespace(cfg=SimpleNamespace(commands=commands)))
        runner.learn=lambda count:setattr(runner,'iteration',runner.iteration+count)
        runner.save=lambda:Path('candidate.pt')
        scope['apply_speed_limit']=lambda *args:None
        scope['evaluate_gate']=lambda args,checkpoint,output,scenario,limit,seed,**kwargs:dict(speed=limit)
        scope['tracking_gate']=lambda tracking,standing,thresholds:(tracking['speed']!=.25,{'steady_speed':tracking['speed']!=.25})
        settings=dict(initial_limit=.6,target_limit=.6,increment=.025,
            chunk_iterations=100,seed=111,target_passes=1,thresholds={},
            additional_tracking_speeds=[.25,.45])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            scope['learn_with_gate'](runner,SimpleNamespace(output_dir=Path(folder)),settings)
            result=SCOPE['json'].loads((Path(folder)/'performance_result.json').read_text())
            self.assertEqual(result['status'],'budget_exhausted')
            gate=SCOPE['json'].loads((Path(folder)/'gate_history.jsonl').read_text())
            self.assertFalse(gate['passed'])
            self.assertFalse(gate['checks']['speed250_steady_speed'])
            self.assertTrue(gate['checks']['speed450_steady_speed'])

    def test_residual_diagnostic_accepts_numpy_estimates_and_zero_command(self):
        scope=dict(SCOPE);scope['__name__']='stability_test_helpers'
        load_code('tools/q1_reward_iteration.py',scope)
        values=np.array([[.27,-.27],[.135,-.135]])
        row=dict(command_m_s=.45)
        scope['record_wheel_residual'](row,values,np.array([True,True]),.3,2.)
        self.assertAlmostEqual(row['wheel_residual_L'],.2025)
        self.assertAlmostEqual(row['wheel_residual_sat_L'],.5)
        row=dict(command_m_s=0.)
        scope['record_wheel_residual'](row,np.zeros((2,2)),np.array([True,False]),.3,2.)
        self.assertEqual(row['wheel_residual_L'],0.)
        self.assertEqual(row['wheel_residual_sat_L'],0.)


if __name__=='__main__':unittest.main()
