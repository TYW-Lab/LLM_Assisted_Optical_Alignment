import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..envs import OptilandAlignmentEnvSimple

class OpticalAlignGym(gym.Env):
    """
    Gymnasium wrapper for OptilandAlignmentEnv.
    - Observation: (7,) float32  (把 NaN/Inf 替换为有限值后再 clip)
    - Action:      (4,) float32 in [-1, 1]  -> 传给 env 作为 (1,4)
    
    注意: simple 版本的环境返回的是单步观测 (7,)，不包含历史信息
    """
    metadata = {"render_modes": []}

    def __init__(self, env_kwargs=None, nan_fill=1e3, seed=None):
        super().__init__()
        env_kwargs = env_kwargs or {}
        self._seed = seed
        self._nan_fill = float(nan_fill)

        # 底层环境
        self.env = OptilandAlignmentEnv(**env_kwargs)

        # 推导空间
        obs_dim, act_dim = self.env.obs_action_space()  # (7, 4)
        obs_max = float(max(self.env.detector_aperture))  # 用探测器口径上限作为观测界
        self.observation_space = spaces.Box(
            low=-obs_max, high=obs_max, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        self._last_info = None

    def _obs_proc(self, obs):
        """处理观测: 展平、处理 NaN/Inf、裁剪到空间范围"""
        obs = np.asarray(obs, dtype=np.float32).flatten()
        
        # 替换 NaN 和 Inf
        obs = np.nan_to_num(obs, nan=self._nan_fill, posinf=self._nan_fill, neginf=-self._nan_fill)
        
        # 裁剪到观测空间范围
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)
        
        return obs

    def reset(self, *, seed=None, options=None):
        """重置环境"""
        if seed is None:
            seed = self._seed
        
        # simple 版本返回 (obs, info)，obs 形状为 (1, 7)
        obs, info = self.env.reset(seed=seed)
        obs = self._obs_proc(obs)
        self._last_info = info
        
        return obs, info

    def step(self, action):
        """执行一步
        
        Args:
            action: (4,) numpy array, 范围 [-1, 1]
        
        Returns:
            obs: (7,) numpy array
            reward: float
            terminated: bool
            truncated: bool
            info: dict
        """
        # Gym 动作为 (4,) -> 环境需要 (1,4)
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        
        # simple 版本返回 (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._obs_proc(obs)
        self._last_info = info
        
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """渲染当前状态"""
        self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()


# ============ 测试代码 ============
if __name__ == "__main__":
    # 配置日志
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
    
    # 创建 Gym 环境
    env_kwargs = {
        'beam_size': 1.0,
        'wavelength': 0.78,
        'detector_aperture': [50.0, 50.0],
        'res': (512, 512),
        'num_rays': 512,
        'max_step': 16,
        'delta_angle_range': 0,
        'log_config': log_config,
        'log_dir': './experiment_logs_wrapper_test',
        'use_wandb': False,
        'wandb_mode': "offline",
        'wandb_project': "optical-alignment-gym-test",
        'wandb_group_name': "wrapper_test",
        'wandb_config': {"algorithm": "random", "experiment": "wrapper-test"},
        'experiment_name': "wrapper_test_run"
    }
    
    env = OpticalAlignGym(env_kwargs=env_kwargs, seed=42)
    
    print("=" * 60)
    print("Gymnasium Wrapper Test")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 运行 2 个 episode
    for ep in range(2):
        obs, info = env.reset(seed=42 + ep)
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}")
        print(f"{'='*60}")
        print(f"Observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        print(f"Initial angles: [{info['init_rx1']:.2f}, {info['init_ry1']:.2f}, "
              f"{info['init_rx2']:.2f}, {info['init_ry2']:.2f}]")
        
        total_reward = 0
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"Step {step}: reward={reward:.2f}, "
                      f"d1={info['distance_pinhole1']:.4f}mm, "
                      f"d2={info['distance_pinhole2']:.4f}mm")
            
            if terminated or truncated:
                print(f"\nEpisode finished at step {step + 1}")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Goal reached: {info['goal_reached']}")
                break
    
    env.close()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)