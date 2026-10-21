"""Viewer controls matching Dreamwaq's four-legged main play key layout."""
import torch


class Q1Keyboard:
    bindings = {
        "KEY_W": "q1_vx_plus", "KEY_S": "q1_vx_minus", "KEY_X": "q1_stop",
        "KEY_SPACE": "q1_pause", "KEY_F": "q1_free_cam",
        "KEY_LEFT_BRACKET": "q1_prev", "KEY_RIGHT_BRACKET": "q1_next",
        "KEY_A": "q1_yaw_plus", "KEY_D": "q1_yaw_minus",
    }

    def __init__(self, env):
        self.env = env
        self.speed = torch.zeros(env.num_envs, device=env.device)
        self.yaw_rate = torch.zeros_like(self.speed)
        self.selected = 0
        self.paused = False
        self.free_camera = False
        self.enabled = False
        self.old_render = env.render
        self.old_handler = getattr(env, "_extra_viewer_action_handler", None)

    def enable(self, gymapi):
        if self.env.viewer is None:
            raise ValueError("Q1 keyboard control requires a viewer; remove --headless")
        for key, action in self.bindings.items():
            self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, getattr(gymapi, key), action)
        for number in range(9):
            self.env.gym.subscribe_viewer_keyboard_event(
                self.env.viewer, getattr(gymapi, "KEY_"+str(number)), "q1_select_"+str(number)
            )
        self.env._extra_viewer_action_handler = self.handle_event
        # The play loop renders BEFORE inference. Suppress the step's second
        # render, so one viewer event queue controls commands and pause.
        self.env.render = lambda *args, **kwargs: None
        self.enabled = True
        print("Click the viewer to focus. W/S: vx +/-0.2 m/s (held after release, limited to +/-0.3); "
              "A/D: yaw +/-0.2 rad/s (limited to +/-0.6); "
              "X: stop translation and turning targets; Space: pause/resume; R: reset selected robot; "
              "F: free/follow camera; 0-8 or [/]: select robot; Esc: exit; V: viewer sync.")
        self.env.gym.subscribe_viewer_keyboard_event(self.env.viewer, gymapi.KEY_R, "q1_reset")

    def handle_event(self, event):
        if self.old_handler is not None:
            self.old_handler(event)
        if event.value <= 0:
            return
        action = event.action
        if action in ("q1_vx_plus", "q1_vx_minus"):
            delta = 0.2 if action == "q1_vx_plus" else -0.2
            self.speed[self.selected] = (self.speed[self.selected]+delta).clamp(-0.3, 0.3)
        elif action in ("q1_yaw_plus", "q1_yaw_minus"):
            delta = 0.2 if action == "q1_yaw_plus" else -0.2
            self.yaw_rate[self.selected] = (self.yaw_rate[self.selected]+delta).clamp(-0.6, 0.6)
        elif action == "q1_stop":
            self.speed[self.selected] = 0.0
            self.yaw_rate[self.selected] = 0.0
        elif action == "q1_pause":
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")
            return
        elif action == "q1_free_cam":
            self.free_camera = not self.free_camera
            if self.free_camera:
                self.env.set_camera(self.env.cfg.viewer.pos, self.env.cfg.viewer.lookat)
            return
        elif action == "q1_prev":
            self.selected = (self.selected-1) % self.env.num_envs
        elif action == "q1_next":
            self.selected = (self.selected+1) % self.env.num_envs
        elif action.startswith("q1_select_"):
            index = int(action.rsplit("_", 1)[1])
            if index >= self.env.num_envs:
                print("Robot {} does not exist (num_envs={})".format(index, self.env.num_envs))
                return
            self.selected = index
        elif action == "q1_reset":
            self.speed[self.selected] = 0.0
            self.yaw_rate[self.selected] = 0.0
            ids = torch.tensor([self.selected], device=self.env.device)
            self.env.reset_idx(ids)
            # Reset fills history, but the cached single frame must match too.
            self.env.obs_buf[ids] = self.env.compute_proprioception_observations()[ids]
        else:
            return
        print("Robot {} target vx={:+.2f} m/s yaw={:+.2f} rad/s".format(
            self.selected, self.speed[self.selected].item(), self.yaw_rate[self.selected].item()))

    def render(self):
        if not self.free_camera:
            target = self.env.root_states[self.selected, :3].detach().cpu().tolist()
            camera = [target[0], target[1]+2.0, target[2]+1.0]
            self.env.set_camera(camera, target)
        self.old_render()

    def apply(self):
        self.env.set_command_target(self.speed, self.yaw_rate)

    def close(self):
        if self.enabled:
            self.env.render = self.old_render
            self.env._extra_viewer_action_handler = self.old_handler
            self.enabled = False
