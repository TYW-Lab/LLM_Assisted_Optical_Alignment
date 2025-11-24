import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from ..utils import visualize_optimization_process
from ..Optical_System.RL_Optical_Sys import OpticalSystem
import cv2

class OpticalAlignmentVisualizer:
    """
    Read data from CSV log files and visualize optical alignment process
    Adapted for new logging structure: ./logs/YYYYMMDD_HHMMSS/
    """
    
    def __init__(self, log_dir='./logs'):
        """
        Initialize visualizer
        
        Args:
            log_dir: Base directory containing timestamped log folders
        """
        self.log_dir = Path(log_dir)
        self.actions_df = None
        self.episodes_df = None
        self.current_exp_dir = None
        
    def list_experiments(self):
        """List all available experiment directories"""
        exp_dirs = sorted([d for d in self.log_dir.iterdir() if d.is_dir()])
        
        if not exp_dirs:
            print(f"No experiment directories found in {self.log_dir}")
            return []
        
        print(f"\n=== Available Experiments ===")
        for i, exp_dir in enumerate(exp_dirs, 1):
            actions_file = exp_dir / 'actions.csv'
            episodes_file = exp_dir / 'episodes.csv'
            
            if actions_file.exists() and episodes_file.exists():
                # Try to count episodes
                try:
                    df = pd.read_csv(episodes_file)
                    n_episodes = len(df)
                    success_rate = df['goal_reached'].mean() * 100
                    print(f"{i}. {exp_dir.name} ({n_episodes} episodes, {success_rate:.1f}% success)")
                except:
                    print(f"{i}. {exp_dir.name}")
        
        return exp_dirs
    
    def load_logs(self, exp_dir=None):
        """
        Load log files from specified experiment directory
        
        Args:
            exp_dir: Experiment directory name (e.g., '20251022_122137')
                     If None, loads the latest experiment
        """
        if exp_dir is None:
            # Load latest experiment
            exp_dirs = sorted([d for d in self.log_dir.iterdir() if d.is_dir()])
            if not exp_dirs:
                raise FileNotFoundError(f"No experiment directories found in {self.log_dir}")
            self.current_exp_dir = exp_dirs[-1]
        else:
            # Load specified experiment
            self.current_exp_dir = self.log_dir / exp_dir
            if not self.current_exp_dir.exists():
                raise FileNotFoundError(f"Experiment directory not found: {self.current_exp_dir}")
        
        # Load CSV files
        actions_file = self.current_exp_dir / 'actions.csv'
        episodes_file = self.current_exp_dir / 'episodes.csv'
        
        if not actions_file.exists() or not episodes_file.exists():
            raise FileNotFoundError(f"Log files not found in {self.current_exp_dir}")
        
        self.actions_df = pd.read_csv(actions_file)
        self.episodes_df = pd.read_csv(episodes_file)
        
        print(f"\nLoaded logs from: {self.current_exp_dir.name}")
        print(f"  Total episodes: {len(self.episodes_df)}")
        print(f"  Total steps: {len(self.actions_df)}")
        print(f"  Episode IDs: {sorted(self.episodes_df['episode_id'].unique().tolist())}")
        
        return self.actions_df, self.episodes_df
    
    def load_latest_logs(self):
        """Load the latest log files (for backward compatibility)"""
        return self.load_logs(exp_dir=None)
    
    def load_specific_logs(self, actions_file, episodes_file):
        """Load specific log files (deprecated, use load_logs instead)"""
        print("Warning: load_specific_logs is deprecated. Use load_logs(exp_dir) instead.")
        self.actions_df = pd.read_csv(actions_file)
        self.episodes_df = pd.read_csv(episodes_file)
        return self.actions_df, self.episodes_df
    
    def _get_visualization_dir(self):
        """Get visualization directory for current experiment"""
        if self.current_exp_dir is None:
            raise ValueError("Please load logs first using load_logs()")
        viz_dir = self.current_exp_dir / 'visualization'
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir
    
    def plot_episode_summary(self, save_path=None):
        """
        Plot summary information for all episodes
        
        Args:
            save_path: Custom save path (if None, saves to experiment's visualization folder)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Steps distribution
        ax = axes[0]
        colors = ['green' if g else 'red' for g in self.episodes_df['goal_reached']]
        ax.bar(self.episodes_df['episode_id'], self.episodes_df['total_steps'], color=colors, alpha=0.6)
        ax.set_xlabel('Episode ID', fontsize=12)
        ax.set_ylabel('Total Steps', fontsize=12)
        ax.set_title('Steps per Episode (Green=Success, Red=Failed)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Success rate and average steps
        ax = axes[1]
        success_rate = self.episodes_df['goal_reached'].mean() * 100
        avg_steps_success = self.episodes_df[self.episodes_df['goal_reached']]['total_steps'].mean()
        avg_steps_failed = self.episodes_df[~self.episodes_df['goal_reached']]['total_steps'].mean()
        
        categories = ['Success Rate (%)', 'Avg Steps\n(Success)', 'Avg Steps\n(Failed)']
        values = [success_rate, avg_steps_success, avg_steps_failed if not np.isnan(avg_steps_failed) else 0]
        colors_bar = ['green', 'blue', 'orange']
        
        bars = ax.bar(categories, values, color=colors_bar, alpha=0.6)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Performance Metrics', fontsize=14)
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Use default path if not provided
        if save_path is None:
            save_path = self._get_visualization_dir() / 'episode_summary.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved episode summary to {save_path}")
        plt.close()
        
        return fig
    
    def plot_episode_trajectory(self, episode_id, save_path=None):
        """
        Plot detailed trajectory for a single episode
        
        Args:
            episode_id: Episode ID to visualize
            save_path: Custom save path (if None, saves to experiment's visualization folder)
        """
        # Get all steps for this episode
        episode_actions = self.actions_df[self.actions_df['episode_id'] == episode_id]
        episode_info = self.episodes_df[self.episodes_df['episode_id'] == episode_id].iloc[0]
        
        if len(episode_actions) == 0:
            print(f"Episode {episode_id} not found!")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)
        
        # 1. Distance evolution (separated)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(episode_actions['step'], episode_actions['distance_pinhole1'], 
                'b-', linewidth=2, label='Pinhole 1 Distance')
        ax1.plot(episode_actions['step'], episode_actions['distance_pinhole2'], 
                'r-', linewidth=2, label='Pinhole 2 Distance')
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Close threshold (0.1mm)')
        ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Success threshold (0.05mm)')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Distance to Target (mm)', fontsize=12)
        ax1.set_title(f'Episode {episode_id} - Distance Evolution', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Cumulative Reward evolution
        ax2 = fig.add_subplot(gs[0, 2])
        cumulative_reward = episode_actions['reward'].cumsum()
        ax2.plot(episode_actions['step'], cumulative_reward, 'r-', linewidth=2)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Cumulative Reward', fontsize=12)
        ax2.set_title('Cumulative Reward Evolution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Angle evolution - Mirror 1
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episode_actions['step'], episode_actions['angle_rx1'], label='RX1', linewidth=2)
        ax3.plot(episode_actions['step'], episode_actions['angle_ry1'], label='RY1', linewidth=2)
        ax3.axhline(y=episode_info['init_rx1'], color='blue', linestyle=':', alpha=0.5)
        ax3.axhline(y=episode_info['init_ry1'], color='orange', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Step', fontsize=12)
        ax3.set_ylabel('Angle (deg)', fontsize=12)
        ax3.set_title('Mirror 1 Angles', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Angle evolution - Mirror 2
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(episode_actions['step'], episode_actions['angle_rx2'], label='RX2', linewidth=2)
        ax4.plot(episode_actions['step'], episode_actions['angle_ry2'], label='RY2', linewidth=2)
        ax4.axhline(y=episode_info['init_rx2'], color='blue', linestyle=':', alpha=0.5)
        ax4.axhline(y=episode_info['init_ry2'], color='orange', linestyle=':', alpha=0.5)
        ax4.set_xlabel('Step', fontsize=12)
        ax4.set_ylabel('Angle (deg)', fontsize=12)
        ax4.set_title('Mirror 2 Angles', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
       # 5. Action components & magnitude
        ax5 = fig.add_subplot(gs[1, 2])

        ax5.plot(episode_actions['step'], episode_actions['action_rx1'], 'r-', label='rx1')
        ax5.plot(episode_actions['step'], episode_actions['action_ry1'], 'r--', label='ry1')
        ax5.plot(episode_actions['step'], episode_actions['action_rx2'], 'b-', label='rx2')
        ax5.plot(episode_actions['step'], episode_actions['action_ry2'], 'b--', label='ry2')

        ax5.set_xlabel('Step', fontsize=12)
        ax5.set_ylabel('Action Value', fontsize=12)
        ax5.set_title('Action', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        
        # 6. Detector 1 displacement trajectory
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(episode_actions['obs_dx1'], episode_actions['obs_dy1'], 
                'b.-', alpha=0.6, linewidth=1, markersize=1)
        ax6.plot(episode_actions['obs_dx1'].iloc[0], episode_actions['obs_dy1'].iloc[0], 
                'go', markersize=5, label='Start')
        ax6.plot(episode_actions['obs_dx1'].iloc[-1], episode_actions['obs_dy1'].iloc[-1], 
                'ro', markersize=5, label='End')
        ax6.plot(0, 0, 'r*', markersize=5, label='Target')

        ax6.set_xlabel('X Displacement (mm)', fontsize=12)
        ax6.set_ylabel('Y Displacement (mm)', fontsize=12)
        ax6.set_title('Detector 1 Trajectory', fontsize=14)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
        
        # 7. Detector 2 displacement trajectory
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(episode_actions['obs_dx2'], episode_actions['obs_dy2'], 
                'b.-', alpha=0.6, linewidth=1, markersize=1)
        ax7.plot(episode_actions['obs_dx2'].iloc[0], episode_actions['obs_dy2'].iloc[0], 
                'go', markersize=5, label='Start')
        ax7.plot(episode_actions['obs_dx2'].iloc[-1], episode_actions['obs_dy2'].iloc[-1], 
                'ro', markersize=5, label='End')
        ax7.plot(0, 0, 'r*', markersize=5, label='Target')

        ax7.set_xlabel('X Displacement (mm)', fontsize=12)
        ax7.set_ylabel('Y Displacement (mm)', fontsize=12)
        ax7.set_title('Detector 2 Trajectory', fontsize=14)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.axis('equal')
        
        # 8. Episode information summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        info_text = f"""
Episode {episode_id} Summary
{'='*30}

Initial State:
Angles (M1): ({episode_info['init_rx1']:.3f}, {episode_info['init_ry1']:.3f})°
Angles (M2): ({episode_info['init_rx2']:.3f}, {episode_info['init_ry2']:.3f})°
Displacement D1: ({episode_info['init_dx1']:.3f}, {episode_info['init_dy1']:.3f}) mm
Displacement D2: ({episode_info['init_dx2']:.3f}, {episode_info['init_dy2']:.3f}) mm

Final State:
Total Steps: {episode_info['total_steps']}
Final Dist P1: {episode_info['final_distance_pinhole1']:.4f} mm
Final Dist P2: {episode_info['final_distance_pinhole2']:.4f} mm
Goal Reached: {'✓ YES' if episode_info['goal_reached'] else '✗ NO'}

Optical System:
M2→P1 Distance: {episode_info['m2_p1_dist']:.2f} mm
P1→P2 Distance: {episode_info['p1_p2_dist']:.2f} mm
"""
        ax8.text(0.1, 0.95, info_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle(f'Episode {episode_id} Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Use default path if not provided
        if save_path is None:
            save_path = self._get_visualization_dir() / f'episode_{episode_id}_trajectory.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved episode trajectory to {save_path}")
        plt.close()
        
        return fig
    
    def visualize_episode_with_optical_system(
        self, 
        episode_id,
        optical_sys_params,
        save_dir=None,
        num_rays=1000,
        res=(256, 256),
        detector_aperture=[20.0, 20.0],
        mirror_radius=12.7,
        aperture_radius=1.0,
        save_every_n_steps=5,
        save_3d=False,
        **viz_kwargs
    ):
        """
        Visualize episode process including optical system irradiance maps
        
        Args:
            episode_id: Episode ID to visualize
            optical_sys_params: Base parameters dictionary for optical system
            save_dir: Custom save directory (if None, saves to experiment's visualization folder)
            num_rays: Number of rays for tracing
            res: Irradiance map resolution
            detector_aperture: Detector aperture
            aperture_radius: Aperture radius
            save_every_n_steps: Save every N steps
            save_3d: Whether to save 3D views
            **viz_kwargs: Additional parameters passed to visualize_optimization_process
        """
        
        # Get episode data
        episode_actions = self.actions_df[self.actions_df['episode_id'] == episode_id]
        episode_info = self.episodes_df[self.episodes_df['episode_id'] == episode_id].iloc[0]
        
        if len(episode_actions) == 0:
            print(f"Episode {episode_id} not found!")
            return
        
        # Determine save path - default to experiment's visualization folder
        if save_dir is None:
            save_path = self._get_visualization_dir() / f'episode_{episode_id}'
        else:
            save_path = Path(save_dir)
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Visualizing episode {episode_id} with optical system...")
        print(f"Total steps: {len(episode_actions)}")
        
        # Create optical system position parameters
        mirror1_pos = np.array([
            episode_info['mirror1_x'],
            episode_info['mirror1_y'],
            episode_info['mirror1_z']
        ])
        mirror2_pos = np.array([
            episode_info['mirror2_x'],
            episode_info['mirror2_y'],
            episode_info['mirror2_z']
        ])
        
        # Build params_log: collect all steps that need visualization
        params_log = []
        for idx, row in episode_actions.iterrows():
            step = row['step']
            
            # Only save specific steps
            if step % save_every_n_steps != 0 and step != len(episode_actions) - 1:
                continue
            
            params_log.append({
                'iteration': step,
                'rx1': row['angle_rx1'],  # Convert to radians
                'ry1': row['angle_ry1'],
                'rx2': row['angle_rx2'],
                'ry2': row['angle_ry2'],
            })
        
        print(f"Prepared {len(params_log)} steps for visualization")
        
        # Create initial optical system
        angles_m1 = np.array([
            episode_info['init_rx1'],
            episode_info['init_ry1'],
            0.0
        ])
        angles_m2 = np.array([
            episode_info['init_rx2'],
            episode_info['init_ry2'],
            0.0
        ])
        
        try:
            optical_sys = OpticalSystem(
                mirror1_position=mirror1_pos,
                mirror2_position=mirror2_pos,
                rotation_angles_mirror1=angles_m1,
                rotation_angles_mirror2=angles_m2,
                m2_a1_dist=episode_info['m2_p1_dist'],
                aperture_dist=episode_info['p1_p2_dist'],
                **optical_sys_params
            )
            
            # Call visualization function
            print(f"Starting visualization with visualize_optimization_process...")
            visualize_optimization_process(
                optical_sys,
                params_log=params_log,
                detector_mirror=3,
                detector_surface1=5,
                detector_surface2=8,
                aperture_radius=aperture_radius,
                mirror_radius=mirror_radius,
                num_rays=num_rays,
                res=res,
                output_base_dir=str(save_path),
                save_3d=save_3d,
                save_irr=True,
                **viz_kwargs
            )
            
            print(f"\nVisualization complete! Saved to {save_path}")
            print(f"\nGenerated files:")
            print(f"  - Irradiance maps: {save_path}/irradiance/")
            if save_3d:
                print(f"  - 3D views: {save_path}/3d_view/")

            # Create videos
            create_video(filepath=f"{save_path}/irradiance/")
            if save_3d:
                create_video(filepath=f"{save_path}/3d_view/")
           
            
        except Exception as e:
            print(f"Error using visualize_optimization_process: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def compare_episodes(self, episode_ids, save_path=None):
        """
        Compare performance of multiple episodes
        
        Args:
            episode_ids: List of episode IDs to compare
            save_path: Custom save path (if None, saves to experiment's visualization folder)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(episode_ids)))
        
        # 1. Distance evolution comparison
        ax = axes[0, 0]
        for ep_id, color in zip(episode_ids, colors):
            ep_data = self.actions_df[self.actions_df['episode_id'] == ep_id]
            total_distance = ep_data['distance_pinhole1'] + ep_data['distance_pinhole2']
            ax.plot(ep_data['step'], total_distance, 
                   label=f'Episode {ep_id}', color=color, linewidth=2, alpha=0.7)
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Success (~0.1)')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Total Distance to Target (mm)', fontsize=12)
        ax.set_title('Distance Evolution Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 2. Cumulative reward comparison
        ax = axes[0, 1]
        for ep_id, color in zip(episode_ids, colors):
            ep_data = self.actions_df[self.actions_df['episode_id'] == ep_id]
            cumulative_reward = ep_data['reward'].cumsum()
            ax.plot(ep_data['step'], cumulative_reward, 
                   label=f'Episode {ep_id}', color=color, linewidth=2, alpha=0.7)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Cumulative Reward', fontsize=12)
        ax.set_title('Cumulative Reward Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Initial vs final distance
        ax = axes[1, 0]
        for ep_id, color in zip(episode_ids, colors):
            ep_info = self.episodes_df[self.episodes_df['episode_id'] == ep_id].iloc[0]
            init_dist = np.sqrt(ep_info['init_dx1']**2 + ep_info['init_dy1']**2) + \
                       np.sqrt(ep_info['init_dx2']**2 + ep_info['init_dy2']**2)
            final_dist = ep_info['final_distance_pinhole1'] + ep_info['final_distance_pinhole2']
            ax.scatter(init_dist, final_dist, 
                      s=200, color=color, alpha=0.6, edgecolors='black', linewidth=2,
                      label=f'Ep {ep_id}')
            ax.annotate(f'{ep_id}', (init_dist, final_dist), 
                       fontsize=10, ha='center', va='center', fontweight='bold')
        ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3)
        ax.set_xlabel('Initial Distance (mm)', fontsize=12)
        ax.set_ylabel('Final Distance (mm)', fontsize=12)
        ax.set_title('Initial vs Final Distance', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        table_data.append(['Episode', 'Steps', 'Final P1', 'Final P2', 'Success', 'Avg Reward'])
        
        for ep_id in episode_ids:
            ep_info = self.episodes_df[self.episodes_df['episode_id'] == ep_id].iloc[0]
            ep_actions = self.actions_df[self.actions_df['episode_id'] == ep_id]
            
            table_data.append([
                str(ep_id),
                str(ep_info['total_steps']),
                f"{ep_info['final_distance_pinhole1']:.4f}",
                f"{ep_info['final_distance_pinhole2']:.4f}",
                '✓' if ep_info['goal_reached'] else '✗',
                f"{ep_actions['reward'].mean():.2f}"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12, 0.12, 0.18, 0.18, 0.12, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Bold header
        for i in range(6):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('Episode Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Use default path if not provided
        if save_path is None:
            save_path = self._get_visualization_dir() / 'episodes_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
        plt.close()
        
        return fig
    
    def get_available_metrics(self):
        """Get all available metrics from the loaded data"""
        if self.actions_df is None or self.episodes_df is None:
            print("Please load data first!")
            return None
        
        metrics = {
            'step_metrics': [col for col in self.actions_df.columns 
                           if col not in ['episode_id', 'step']],
            'episode_metrics': [col for col in self.episodes_df.columns 
                              if col not in ['episode_id']]
        }
        
        print("\n=== Available Metrics ===")
        print("\nStep-level metrics (from actions.csv):")
        for m in metrics['step_metrics']:
            print(f"  - {m}")
        
        print("\nEpisode-level metrics (from episodes.csv):")
        for m in metrics['episode_metrics']:
            print(f"  - {m}")
        
        return metrics

def extract_num(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1

def create_video(filepath=""):
    image_folder = filepath
    video_dir = os.path.join(filepath, "video")
    os.makedirs(video_dir, exist_ok=True)
    video_name = os.path.join(video_dir, "output.mp4")
    
    fps = 10

    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    images.sort(key=extract_num)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)

    video.release()
    print(f"Video Created: {video_name}")

# ============ Usage Example ============
if __name__ == "__main__":
    # Create visualizer
    viz = OpticalAlignmentVisualizer(log_dir='./experiment_logs_test')
    
    # List available experiments
    viz.list_experiments()
    
    # Load latest logs (or specify experiment: viz.load_logs('20251022_122137'))
    actions_df, episodes_df = viz.load_latest_logs()
    # actions_df, episodes_df = viz.load_logs('20251028_235056_1')

    
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
        'pinhole_aperture': [1.0, 0.0],
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