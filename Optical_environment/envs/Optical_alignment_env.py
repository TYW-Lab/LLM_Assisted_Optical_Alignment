import numpy as np
import torch
from ..Optical_System.RL_Optical_Sys import OpticalSystem
from ..utils import ExperimentLogger, compute_irr, spot_centroid_from_irr, get_detector_edges, compute_power
from ..tools.visualize import *
"""
This version does not limit the accumulated action magnitude, allowing for larger angle adjustments.

And the reset logic is:
1. Randomly pertub delta parameters within specified ranges.
2. Create OpticalSystem.
3. Add a random action to this initial setup to avoid starting from a perfect alignment.
(It's different from return to base angles and then add random actions)
"""

class OptilandAlignmentEnv:
    """
    Optiland-based optical alignment environment.
    Observation: spot displacements on two detectors [m2dx, m2dy, hit_m2, p1dx, p1dy, p2dx, p2dy, 
                                                      action1_valid, action2_valid,
                                                      action3_valid, action4_valid]
    >> Total observation is Pre_observation + Now_observation 
                                                
    Action: four mirror angle increments [delta_rx1, delta_ry1, delta_rx2, delta_ry2]
    All the unit is milimeter (mm)
    """
    
    def __init__(
            self,
        beam_size=0.5,
        wavelength=0.78,
        mirror_aperture=[12.7, 0.0],
        pinhole_aperture=[0.5, 0.0],
        detector_aperture=[50.0, 50.0],
        mirror1_position=np.array([0.0, 0.0, 100.0]),
        mirror2_position=np.array([0.0, -100.0, 100.0]),
        rotation_angles_mirror1=np.array([np.pi-np.pi/4, 0.0, 0.0]),
        rotation_angles_mirror2=np.array([-np.pi/4, 0.0, 0.0]),
        m2_a1_dist=70.0,
        p1_p2_dist=80.0,
        mirror2_detector=3,
        pinhole1_detector=5,
        pinhole2_detector=8,
        change_pinhole_size=False,
        device='cpu',
        max_step=512,
        res=(512, 512),
        num_rays=400,
        delta_angle_range=2.4*np.pi/180, #2.4
        delta_mirror2_position=0, #10
        delta_pinhole_distance=0, #20
        target_position=(0.0, 0.0),
        threshold=0.95, # Power threshold for success
        log_every_n_episodes=1,  # log every N episodes
        log_every_n_steps=1,  # log every N steps
        log_dir='./logs',
        log_config = None,
        use_wandb=False,
        wandb_mode="offline",  # "online", "offline", or "disabled"
        wandb_project=None,
        wandb_entity=None,
        wandb_config=None,
        wandb_run_name=None,
        wandb_group_name=None,
        experiment_name = None
    ):
        self.optical_params = {
            'beam_size': beam_size,
            'wavelength': wavelength,
            'mirror_aperture': mirror_aperture,
            'pinhole_aperture': pinhole_aperture,
            'detector_aperture': detector_aperture,
        }
        # Basic parameters
        self.beam_size = beam_size
        self.wavelength = wavelength
        self.mirror_aperture = mirror_aperture
        self.pinhole_aperture = pinhole_aperture
        self.detector_aperture = detector_aperture
        self.device = device
        self.max_step = max_step
        self.res = res
        self.num_rays = num_rays
        self.angle_range = delta_angle_range
        self.delta_mirror2_position_range = delta_mirror2_position
        self.delta_pinhole_distance = delta_pinhole_distance
        self.target_position = np.array(target_position, dtype=np.float32)
        self.threshold = threshold
        
        self.mirror2_detector = mirror2_detector
        self.pinhole1_detector = pinhole1_detector
        self.pinhole2_detector = pinhole2_detector
        # Default positions/angles
        self.mirror1_position_base = np.array(mirror1_position)
        self.mirror2_position_base = np.array(mirror2_position)
        self.rotation_angles_mirror1_base = np.array(rotation_angles_mirror1)
        self.rotation_angles_mirror2_base = np.array(rotation_angles_mirror2)
        
        self.m2_a1_dist = m2_a1_dist
        self.p1_p2_dist = p1_p2_dist

        # Get detector edges
        self.x_edges, self.y_edges = get_detector_edges(detector_aperture, self.res)
        self.ideal_power_1 = 0.0
        self.ideal_power_2 = 0.0

        # State variables
        self.T = 0
        self.count = 0
        self.acc_reward = 0.0
        self.optical_sys = None
        self.current_angles = None
        self.pass_p1 = None
        self.last_displacement = np.zeros((1, 7), dtype=np.float32)
        self.action_acc = []
        self.dead = False
        
        # Environment bounds
        self.change_pinhole = change_pinhole_size # True to enlarge pinhole 1 size during observation
        max_displacement = max(detector_aperture)
        self.obs_low = -max_displacement
        self.obs_high = max_displacement
        self.action_low = -1.0
        self.action_high = 1.0
        self.previous_obs = None

        self.use_wandb = use_wandb
        # Init logger
        env_config = {
            'beam_size': beam_size,
            'wavelength': wavelength,
            'max_step': max_step,
            'num_rays': num_rays,
            'delta_angle_range': delta_angle_range,
            'target_position': target_position,
            'res': res,
        }
        
        if wandb_config:
            env_config.update(wandb_config)

        log_config = self.default_log() if log_config is None else log_config
        self.logger = ExperimentLogger(
            log_types=log_config,
            log_dir=log_dir,
            experiment_name=experiment_name,
            use_wandb=use_wandb,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_config=wandb_config,
            wandb_run_name=wandb_run_name,
            wandb_group_name=wandb_group_name,
            auto_flush=False,
        )
        self.log_dir = log_dir
        self.current_episode_id = 0
        self.episode_info = None
        self.step_info = None
        self.log_every_n_episodes=log_every_n_episodes  # log every N episodes
        self.log_every_n_steps=log_every_n_steps  # log every N steps

    def _create_optical_system(self, mirror1_pos, mirror2_pos, angles_m1, angles_m2, m2_p1_dist, p1_p2_dist):
        """Create the optical system instance."""
        try:
            optical_sys = OpticalSystem(
                beam_size=self.beam_size,
                wavelength=self.wavelength,
                mirror_aperture=self.mirror_aperture,
                pinhole_aperture=self.pinhole_aperture,
                detector_aperture=self.detector_aperture,
                mirror1_position=mirror1_pos,
                mirror2_position=mirror2_pos,
                rotation_angles_mirror1=angles_m1,
                rotation_angles_mirror2=angles_m2,
                m2_a1_dist=m2_p1_dist,
                aperture_dist=p1_p2_dist,
            )
            
            if not optical_sys.is_valid:
                return False
            
            self.optical_sys = optical_sys
            self.ideal_power_1 = compute_power(
                self.optical_sys,
                detector_surface=self.pinhole1_detector,
                res=self.res,
                num_rays=self.num_rays
            )
            self.ideal_power_2 = compute_power(
                self.optical_sys,
                detector_surface=self.pinhole2_detector,
                res=self.res,
                num_rays=self.num_rays
            )
            return True
        except Exception as e:
            print(f"Failed to create optical system: {e}")
            return False
    
    def _get_centroid_displacement(self, detector_surface):
        """Get the spot centroid displacement relative to the target position."""
        try:
            irr_map = compute_irr(
                self.optical_sys,
                detector_surface=detector_surface,
                res=self.res,
                num_rays=self.num_rays,
                normalize=True
            )
            
            xc, yc, M0 = spot_centroid_from_irr(
                irr_map,
                self.x_edges,
                self.y_edges,
                tau=0.05,
                eps=1e-12,
                res=self.res,
                negate=False
            )
            
            if torch.is_tensor(xc):
                xc = xc.item()
            if torch.is_tensor(yc):
                yc = yc.item()
            if torch.is_tensor(M0):
                M0 = M0.item()
            
            if M0 < 1e-6:
                return np.nan, np.nan
            
            delta_x = xc - self.target_position[0]
            delta_y = yc - self.target_position[1]
        
            return delta_x, delta_y
            
        except Exception as e:
            print(f"Centroid computation failed (detector {detector_surface}): {e}")
            return self.detector_aperture[0], self.detector_aperture[1]
    
    def _get_observation(self):
        """Get the current observation."""
        mirror2_dx, mirror2_dy = self._get_centroid_displacement(detector_surface=self.mirror2_detector)
        dist_beam_to_mirror2 = np.sqrt(mirror2_dx**2 + mirror2_dy**2) if not np.isnan(mirror2_dx) and not np.isnan(mirror2_dy) else np.nan
        
        if dist_beam_to_mirror2 > self.mirror_aperture[0] or (np.isnan(mirror2_dx) or np.isnan(mirror2_dy)):
            obs = np.array([0.0, np.nan, np.nan, np.nan, np.nan])
        else:
            # Enlarge pinhole 1 size to max in order to increase space of observation
            self.optical_sys.set_pinhole_size(enlarge=self.change_pinhole)
            pinhole1_dx, pinhole1_dy = self._get_centroid_displacement(detector_surface=self.pinhole1_detector)
            pinhole2_dx, pinhole2_dy = self._get_centroid_displacement(detector_surface=self.pinhole2_detector)
            # Change pinhole 1 to original size
            self.optical_sys.set_pinhole_size(enlarge=False)

            obs = np.array([1.0, pinhole1_dx, pinhole1_dy, pinhole2_dx, pinhole2_dy], dtype=np.float32)
            obs = np.clip(obs, self.obs_low, self.obs_high)

        obs = np.append([mirror2_dx, mirror2_dy], obs)
        obs = np.expand_dims(obs, axis=0)
        self.last_displacement = obs.copy()  # shape (1,7)

        return obs

    def _compute_reward(self, action, obs):
        """Compute the reward from the current observation."""
        reward_mirror_angles = -((np.abs(action) > 0.98) * 0.5).sum() 
        reward_pinhole_1_pass = 0.6 if self.goal(pinhole_index=1) else 0.0
        reward_pinhole_2_pass = 300 if self.goal(pinhole_index=2) else 0.0
        reward_steps = -80 if self.T == self.max_step else 0.0

        reward = reward_mirror_angles + reward_pinhole_1_pass + reward_pinhole_2_pass + reward_steps - 1.0

        return reward, 0
    
    def _is_terminated(self, obs):
        """Check termination conditions."""
        if self.goal(pinhole_index=1) and self.goal(pinhole_index=2):
            return True
        return False
    
    def reset(self, seed=None):
        """Reset environment and start new episode."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        conf_type = np.random.randint(0,3,1).squeeze()
        self.select_configuration(conf_type=conf_type) # random select from 4 valid configuration
        
        create_success = False
        while not create_success:
            self.optical_sys = None
            delta_x = np.random.uniform(-self.delta_mirror2_position_range, self.delta_mirror2_position_range)
            delta_y = np.random.uniform(-self.delta_mirror2_position_range, self.delta_mirror2_position_range)
            delta_z = np.random.uniform(-self.delta_mirror2_position_range, self.delta_mirror2_position_range)
            
            mirror1_pos = self.mirror1_position_base.copy()
            mirror2_pos = self.mirror2_position_base.copy()
            mirror2_pos[0] += delta_x
            mirror2_pos[1] += delta_y
            mirror2_pos[2] += delta_z
            
            rx1 = self.rotation_angles_mirror1_base[0] + np.random.uniform(-self.angle_range, self.angle_range)
            ry1 = self.rotation_angles_mirror1_base[1] + np.random.uniform(-self.angle_range, self.angle_range)
            rx2 = self.rotation_angles_mirror2_base[0] + np.random.uniform(-self.angle_range, self.angle_range)
            ry2 = self.rotation_angles_mirror2_base[1] + np.random.uniform(-self.angle_range, self.angle_range)
            
            angles_m1 = np.array([rx1, ry1, 0.0])
            angles_m2 = np.array([rx2, ry2, 0.0])
            
            delta_z1 = np.random.uniform(-self.delta_pinhole_distance, self.delta_pinhole_distance)
            delta_z2 = np.random.uniform(-self.delta_pinhole_distance, self.delta_pinhole_distance)
            
            m2_a1_dist = self.m2_a1_dist + delta_z1
            p1_p2_dist = self.p1_p2_dist + delta_z2 - delta_z1
            
            create_success = self._create_optical_system(
                mirror1_pos, mirror2_pos, angles_m1, angles_m2, m2_a1_dist, p1_p2_dist
            )
        initial_angles = [[rx1,ry1,rx2,ry2]]
        
        # Reset envirenment
        rx1, ry1 = self.rotation_angles_mirror1_base[0], self.rotation_angles_mirror1_base[1]
        rx2, ry2 = self.rotation_angles_mirror2_base[0], self.rotation_angles_mirror2_base[1]
        self.optical_sys.set_mirror_angle(rx1, ry1, rx2, ry2)
        self.current_angles = np.array([[rx1, ry1, rx2, ry2]], dtype=np.float32)
        self.T = 0
        self.count = 0
        self.acc_reward = 0.0
        self.current_episode_id += 1

        # Random initial action to initialize system
        action = 0.1 * (2 * np.random.rand(1, 4) - 1).astype(np.float32)
        self.action_acc = action
        angles = np.deg2rad(action * 4) + self.current_angles

        self.optical_sys.set_mirror_angle(
            rx1=angles[0][0],
            ry1=angles[0][1],
            rx2=angles[0][2],
            ry2=angles[0][3],
            gradient=False
        )
        current_angles = self.optical_sys.get_mirror_angle()
        self.current_angles = [current_angles]

        # Get initial observation
        obs = self._get_observation().astype(np.float32)
        act_mask = np.where(np.abs(self.action_acc) < 1.0, 0.0, self.action_acc)
        obs = np.concatenate((obs, act_mask), axis=1)
        self.previous_obs = obs


        self.episode_info = {
            "episode_id":self.current_episode_id,
            "init_rx1":initial_angles[0][0], "init_ry1":initial_angles[0][1],
            "init_rx2":initial_angles[0][2], "init_ry2":initial_angles[0][3],            
            "mirror1_x":mirror1_pos[0], "mirror1_y":mirror1_pos[1], "mirror1_z":mirror1_pos[2],
            "mirror2_x":mirror2_pos[0], "mirror2_y":mirror2_pos[1], "mirror2_z":mirror2_pos[2],
            "m2_p1_dist": m2_a1_dist, "p1_p2_dist": p1_p2_dist,
            "init_dx1":obs[0][3], "init_dy1":obs[0][4], "init_dx2":obs[0][5], "init_dy2":obs[0][6],
        }
        
        
        self.step_info = {
            "episode_id": self.current_episode_id,
            "step": self.T,
            # Actions (increments)
            "action_rx1": self.action_acc[0][0],
            "action_ry1": self.action_acc[0][1],
            "action_rx2": self.action_acc[0][2],
            "action_ry2": self.action_acc[0][3],
            # Current angles (absolute values)
            "angle_rx1": self.current_angles[0][0],
            "angle_ry1": self.current_angles[0][1],
            "angle_rx2": self.current_angles[0][2],
            "angle_ry2": self.current_angles[0][3],
            # pinhole observation
            'obs_dx1': obs[0][3],
            'obs_dy1': obs[0][4],
            'obs_dx2': obs[0][5],
            'obs_dy2': obs[0][6],
            # Reward
            "reward": 0,
            "distance_pinhole1": np.sqrt(obs[0][3]**2 + obs[0][4]**2),
            "distance_pinhole2": np.sqrt(obs[0][5]**2 + obs[0][6]**2),
            'terminated': False,
            'truncated': False,
        }
        # step == 0
        action = 0.1 * (2 * np.random.rand(1, 4) - 1).astype(np.float32)
        total_obs, _, _, _, _ = self.step(action, first_step = True)
        return total_obs
    
    def step(self, action, first_step=False):
        """Take one environment step and log the action."""
        # action range:[-4,4] degrees
        action_ = np.copy(action).astype(np.float32)
        #action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        #delta_angles = np.deg2rad(action * 4) * 0.1
        self.action_acc = np.clip(self.action_acc + action, self.action_low, self.action_high)

        rx1 = np.deg2rad(self.action_acc[0][0] * 4) + self.rotation_angles_mirror1_base[0]
        ry1 = np.deg2rad(self.action_acc[0][1] * 4) + self.rotation_angles_mirror1_base[1]
        rx2 = np.deg2rad(self.action_acc[0][2] * 4) + self.rotation_angles_mirror2_base[0]
        ry2 = np.deg2rad(self.action_acc[0][3] * 4) + self.rotation_angles_mirror2_base[1]
        
        # Update optical system with new angles
        self.optical_sys.set_mirror_angle(rx1=rx1,ry1=ry1,rx2=rx2,ry2=ry2,gradient=False)
        current_angles = self.optical_sys.get_mirror_angle()
        self.current_angles = [current_angles]
        
        obs = self._get_observation().astype(np.float32)
        act_mask = np.where(np.abs(self.action_acc) < 1.0, 0.0, self.action_acc)

        obs_ = np.concatenate((obs, act_mask), axis=1)
        total_obs = np.concatenate((self.previous_obs, obs_, action_), axis=1)
        self.previous_obs = obs_
        self.T += 1 if not first_step else 0
        reward, rs = self._compute_reward(action, obs)
    
        self.acc_reward += reward
        terminated = self._is_terminated(obs)
        truncated = (self.T >= self.max_step)
        
        self.step_info = {
            "episode_id": self.current_episode_id,
            "step": self.T,
            # Actions (increments)
            "action_rx1": self.action_acc[0][0],
            "action_ry1": self.action_acc[0][1],
            "action_rx2": self.action_acc[0][2],
            "action_ry2": self.action_acc[0][3],
            # Current angles (absolute values)
            "angle_rx1": self.current_angles[0][0],
            "angle_ry1": self.current_angles[0][1],
            "angle_rx2": self.current_angles[0][2],
            "angle_ry2": self.current_angles[0][3],
            # pinhole observation
            'obs_dx1': obs[0][3],
            'obs_dy1': obs[0][4],
            'obs_dx2': obs[0][5],
            'obs_dy2': obs[0][6],
            # Reward
            "reward": reward,
            "distance_pinhole1": np.sqrt(obs[0][3]**2 + obs[0][4]**2),
            "distance_pinhole2": np.sqrt(obs[0][5]**2 + obs[0][6]**2),
            'terminated':terminated,
            'truncated':truncated,            
        }

        if self.current_episode_id % self.log_every_n_episodes == 0:
            if self.T % self.log_every_n_steps == 0:
                self.logger.log("actions",self.step_info,save_wandb=self.use_wandb)
            if terminated or truncated:
                self.episode_info["total_steps"] = self.T
                self.episode_info["final_distance_pinhole1"] = np.sqrt(obs[0][3]**2 + obs[0][4]**2)
                self.episode_info["final_distance_pinhole2"] = np.sqrt(obs[0][5]**2 + obs[0][6]**2)
                self.episode_info["goal_reached"] = terminated and not truncated
                self.logger.log("episodes",self.episode_info,self.use_wandb)
                self.logger.flush_all()
                self.logger.finish_episode()
        
        return total_obs, reward, rs, terminated, truncated
    
    def goal(self, pinhole_index=1):
        if pinhole_index == 1:
            power = compute_power(
                self.optical_sys,
                detector_surface=self.pinhole1_detector,
                res=self.res,
                num_rays=self.num_rays
            )
            return power > (self.ideal_power_1 * self.threshold)
        elif pinhole_index == 2:
            power = compute_power(
                self.optical_sys,
                detector_surface=self.pinhole2_detector,
                res=self.res,
                num_rays=self.num_rays
            )
            return power > (self.ideal_power_2 * self.threshold)
        
    def obs_action_space(self):
        """Return observation and action space dimensions."""
        return 26, 4
    
    def render(self, episode_id=1, episode_summary=True, episode_trajectory=True,
               log_file=None, save_every_n_steps=1, save_3d=True, res=(512, 512)):
        """Render the optical system through log"""
        # Create visualizer
        viz = OpticalAlignmentVisualizer(log_dir=self.log_dir)
        
        # List available experiments
        viz.list_experiments()
        
        if log_file is not None:
            actions_df, episodes_df = viz.load_logs(log_file)
        else:
            actions_df, episodes_df = viz.load_latest_logs()

        if episode_summary is True:
            viz.plot_episode_summary()

        if episode_trajectory is True:
            viz.plot_episode_trajectory(episode_id=1)
        
        viz.visualize_episode_with_optical_system(
            episode_id=episode_id,
            optical_sys_params=self.optical_params,
            save_every_n_steps=save_every_n_steps,  # Save every step
            num_rays=self.num_rays,
            save_3d=save_3d,
            aperture_radius=self.optical_params['pinhole_aperture'][0],
            mirror_radius=self.optical_params['mirror_aperture'][0],
            res=res
        )
    
    def select_configuration(self, conf_type=0):
        """Select one of the four valid configurations."""
        conf_type = 0
        M1_angle = [180-45, 180-45, 180+45, 180+45]
        M2_angle = [-45, 180+45, 360+45, 180-45]
        self.rotation_angles_mirror1_base = np.array([M1_angle[conf_type], 0.0, 0.0]) * np.pi / 180.0
        self.rotation_angles_mirror2_base = np.array([M2_angle[conf_type], 0.0, 0.0]) * np.pi / 180.0
        #mirror1_position = np.array([0.0, 0.0, 100.0]),
        M2_pos = [-100.0, -100.0, 100.0, 100.0]
        #self.mirror1_position_base = np.array(mirror1_position)
        self.mirror2_position_base = np.array([0.0, M2_pos[conf_type], 100.0])
        return
    def print_status(self):
        """Render the current state to stdout."""
        print(f"Step {self.T}:")
        print(f"  Angles (deg): {np.rad2deg(self.current_angles)}")
        print(f"  Distance mirror2: {np.sqrt(self.last_displacement[0][0]**2 + self.last_displacement[0][1]**2):.4f} mm")
        print(f"  Distance pinhole1: {np.sqrt(self.last_displacement[0][3]**2 + self.last_displacement[0][4]**2):.4f} mm")
        print(f"  Distance pinhole2: {np.sqrt(self.last_displacement[0][5]**2 + self.last_displacement[0][6]**2):.4f} mm")

    def close(self):
        """Clean up resources."""
        self.logger.close()
        self.optical_sys = None


# ============ Example usage ============
if __name__ == "__main__":

    log_config = {
            "actions": [
                "episode_id", "step",
                "action_rx1", "action_ry1", "action_rx2", "action_ry2",
                "angle_rx1", "angle_ry1", "angle_rx2", "angle_ry2",
                "obs_dx1", "obs_dy1", "obs_dx2", "obs_dy2",
                "reward", "distance_pinhole1", "distance_pinhole2",
                "terminated", "truncated"
            ],
            "episodes": [
                "episode_id",
                "init_rx1", "init_ry1", "init_rx2", "init_ry2",
                "init_dx1", "init_dy1", "init_dx2", "init_dy2",
                "mirror1_x", "mirror1_y", "mirror1_z",
                "mirror2_x", "mirror2_y", "mirror2_z",
                "m2_p1_dist", "p1_p2_dist",
                "total_steps", "goal_reached",
                "final_distance_pinhole1", "final_distance_pinhole2"
            ]
        }
    
    wandb_project = "optical-alignment-demo"
    env = OptilandAlignmentEnv(
        beam_size=1.0,  # r mm
        wavelength=0.78,
        detector_aperture=[50.0, 50.0],
        res=(512, 512),
        num_rays=512,
        max_step=16,
        delta_angle_range=2.4,
        log_config=log_config,
        log_dir='./experiment_logs_test',
        use_wandb=False,
        wandb_mode="offline",  # Save logs locally, sync later
        wandb_project=wandb_project,
        wandb_group_name="test5",
        wandb_config={"algorithm": "random", "experiment": "offline-mode"},
    )
    
    print("=" * 50)
    print("Environment Test with Logger")
    print("=" * 50)
    
    # Run 3 episodes
    for ep in range(3):
        obs = env.reset(seed=42+ep)
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}")
        print(f"{'='*50}")
        # print(f"Initial angles: {info['angles']}")
        
        for step in range(600):
            action = np.random.uniform(-0.4, 0.4, size=(1, 4)).astype(np.float32)
            obs, reward, rs, terminated, truncated = env.step(action)
            
            # print(f"Step {step+1}: pinhole1={info['distance_pinhole1']:.4f} mm, "
            #       f"pinhole2={info['distance_pinhole2']:.4f} mm, reward={reward:.2f}")
            env.render()
            
            if terminated:
                print("Aligned successfully!")
                break
            if truncated:
                print("Reached max steps")
                break
    
    env.close()