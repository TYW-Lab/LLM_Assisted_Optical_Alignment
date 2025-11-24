from Optical_environment.tools import OpticalAlignmentVisualizer
# Create visualizer
viz = OpticalAlignmentVisualizer(log_dir='./experiment_logs')

# List available experiments
viz.list_experiments()

# Load latest logs (or specify experiment: viz.load_logs('20251022_122137'))
actions_df, episodes_df = viz.load_latest_logs()
# actions_df, episodes_df = viz.load_logs('20251118_163512')


# All visualizations will now be saved in: ./logs/YYYYMMDD_HHMMSS/visualization/

# 1. Plot summary of all episodes (auto-saved to visualization folder)
viz.plot_episode_summary()

# 2. Plot detailed trajectory of a single episode (auto-saved)
viz.plot_episode_trajectory(episode_id=1)

# 3. Compare multiple episodes (auto-saved)
# viz.compare_episodes(episode_ids=[1, 2, 3])

# 4. Visualize optical system irradiance (auto-saved to visualization/episode_N/)
optical_params = {
    'beam_size': 1.0,
    'wavelength': 0.78,
    'mirror_aperture': [12.7, 0.0],
    'pinhole_aperture': [1.5, 0.0],
    'detector_aperture': [50.0, 50.0],
}

viz.visualize_episode_with_optical_system(
    episode_id=1,
    optical_sys_params=optical_params,
    save_every_n_steps=1,  # Save every step
    num_rays=1000,
    save_3d=True,
    aperture_radius=1.0,
    mirror_radius=12.7,
    res=(512, 512)
)

# You can still specify custom paths if needed:
# viz.plot_episode_summary(save_path='./custom/path/summary.png')
# viz.visualize_episode_with_optical_system(..., save_dir='./custom/path/')

print("\n" + "="*50)
print("All visualizations saved!")
print(f"Location: {viz.current_exp_dir / 'visualization'}")
print("="*50)