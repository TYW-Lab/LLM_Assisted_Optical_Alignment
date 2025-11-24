log_dir='./experiment_logs'

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
use_wandb = False
wandb_mode="offline"  # Save logs locally, sync later
wandb_group_name="test5"
wandb_config={"algorithm": "random", "experiment": "offline-mode"}