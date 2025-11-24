import csv
import os
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import numpy as np
import wandb

class ExperimentLogger:
    """Generic experiment logger supporting multiple log types and custom columns"""
    
    def __init__(
        self,
        log_types: Dict[str, List[str]],
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_mode: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_run_name: Optional[str] = None,
        wandb_group_name: Optional[str] = None,
        auto_flush: bool = True,
    ):
        """
        Initialize generic experiment logger
        
        Args:
            log_types: Log type configuration dict, format: {"log_type_name": ["column1", "column2", ...]}
                      Example: {"training": ["epoch", "loss", "accuracy"], 
                               "system": ["param1", "param2"]}
            log_dir: Directory for saving logs
            experiment_name: Experiment name (uses timestamp if None)
            use_wandb: Whether to use WandB
            wandb_mode: WandB mode - "online" (real-time upload), "offline" (local save), "disabled"
            wandb_project: WandB project name
            wandb_entity: WandB entity name
            wandb_config: WandB configuration dictionary
            wandb_run_name: WandB run name
            wandb_group_name: WandB group name
            auto_flush: Whether to auto-flush to file (write immediately on each log)
        """
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.auto_flush = auto_flush
        
        # Validate log_types format
        if not isinstance(log_types, dict) or not log_types:
            raise ValueError("log_types must be a non-empty dictionary")
        for log_type, columns in log_types.items():
            if not isinstance(columns, list) or not columns:
                raise ValueError(f"Columns for '{log_type}' must be a non-empty list")
        
        self.log_types = log_types
        
        # WandB configuration
        if use_wandb and wandb_mode is None:
            wandb_mode = "online"
        self.wandb_mode = wandb_mode.lower() if (use_wandb and wandb_mode) else "disabled"
        
        if self.wandb_mode not in ["online", "offline", "disabled"]:
            raise ValueError(f"wandb_mode must be 'online', 'offline', or 'disabled', got '{wandb_mode}'")
        
        if self.use_wandb and self.wandb_mode != "disabled":
            os.environ["WANDB_MODE"] = self.wandb_mode
        
        # Timestamp and experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp
        self.experiment_name = os.path.join(experiment_name, timestamp) if experiment_name is not None else f"{timestamp}"
        
        # Data cache: create cache for each log type
        self.data_cache: Dict[str, List[List[Any]]] = {
            log_type: [] for log_type in log_types.keys()
        }
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        self.exp_log_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        
        # Create CSV file for each log type
        self.csv_paths: Dict[str, str] = {}
        self._init_csv_files()
        
        # WandB configuration
        self.wandb_run = None
        self.wandb_project = wandb_project or "experiments"
        self.wandb_entity = wandb_entity
        self.wandb_config = wandb_config or {}
        self.wandb_group_name = wandb_group_name or wandb_run_name
        self._base_run_name = wandb_run_name
        self.wandb_table = None

        # Create WandB directory
        self._exp_dir = None
        if self.use_wandb and self.wandb_mode != "disabled":
            self._exp_dir = os.path.join("wandb_log", self.wandb_group_name)
            os.makedirs(self._exp_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = self._exp_dir
            self._start_wandb_run()
        
        # Print initialization info
        self._print_init_info()
    
    def _init_csv_files(self):
        """Initialize CSV files and write headers for each log type"""
        for log_type, columns in self.log_types.items():
            csv_path = os.path.join(self.exp_log_dir, f"{log_type}.csv")
            self.csv_paths[log_type] = csv_path
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
    
    def _print_init_info(self):
        """Print initialization information"""
        print(f"\n{'='*60}")
        print(f"Experiment Logger Initialized: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Log directory: {self.exp_log_dir}")
        print(f"Log types: {list(self.log_types.keys())}")
        
        if self.use_wandb and self.wandb_mode != "disabled":
            mode_str = "OFFLINE" if self.wandb_mode == "offline" else "ONLINE"
            print(f"WandB {mode_str} mode enabled")
            print(f"  Project: {self.wandb_project}")
            print(f"  Group: {self.wandb_group_name}")
            if self.wandb_mode == "offline":
                print(f"  Local logs: ./wandb_log/{self.wandb_group_name}/wandb")
                print(f"  Sync command: wandb sync ./wandb_log/{self.wandb_group_name}/wandb/offline-*")
        else:
            print("WandB disabled")
        print(f"{'='*60}\n")
    
    def log(self, log_type: str, data: Union[List, Dict], save_wandb: str = False):
        """
        Log a single entry
        
        Args:
            log_type: Log type (must be defined during initialization)
            data: Log data, can be list or dict
                 - If list: must match the number of columns for this log_type
                 - If dict: keys must match column names
        """
        if log_type not in self.log_types:
            raise ValueError(f"Unknown log_type '{log_type}'. Available types: {list(self.log_types.keys())}")
        
        columns = self.log_types[log_type]
        
        # Convert data format
        if isinstance(data, dict):
            # Dictionary format: extract values in column order
            row_data = [data.get(col, None) for col in columns]
        elif isinstance(data, (list, tuple)):
            # List format: use directly
            if len(data) != len(columns):
                raise ValueError(
                    f"Data length ({len(data)}) doesn't match columns ({len(columns)}) for log_type '{log_type}'"
                )
            row_data = list(data)
        else:
            raise ValueError(f"Data must be a list, tuple, or dict, got {type(data)}")
        
        # If auto_flush is enabled, write to CSV immediately
        if not self.auto_flush:
            self.data_cache[log_type].append(row_data)
        elif  self.auto_flush:
            with open(self.csv_paths[log_type], 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        
        # Log to WandB
        if self.use_wandb and self.wandb_mode != "disabled" and self.wandb_run is not None and save_wandb:
            self._log_to_wandb(log_type, columns, row_data)
    
    def _log_to_wandb(self, log_type: str, columns: List[str], row_data: List[Any]):
        """Log data to WandB"""
        try:
            # Create log dict with log_type prefix to avoid conflicts
            log_dict = {f"{log_type}/{col}": val for col, val in zip(columns, row_data)}
            wandb.log(log_dict)
        except Exception as e:
            print(f"Warning: Failed to log to WandB: {e}")
    
    def log_wandb_table(self, log_type: str, table_data: Union[List, Dict], table_name: Optional[str] = None):
        """
        Log table data to WandB
        
        Args:
            log_type: Log type
            table_data: Table data (2D list)
            table_name: Table name (defaults to log_type)
        """
        if not (self.use_wandb and self.wandb_mode != "disabled" and self.wandb_run is not None):
            return
        
        if log_type not in self.log_types:
            raise ValueError(f"Unknown log_type '{log_type}'")
        
        if self.wandb_table is None:
             columns = self.log_types[log_type]
             self.wandb_table = wandb.Table(columns=columns)
        try:
                # Convert data format
            if isinstance(table_data, dict):
                row_data = [table_data.get(col, None) for col in columns]
            elif isinstance(table_data, (list, tuple)):
                if len(table_data) != len(columns):
                    raise ValueError(
                        f"Data length ({len(table_data)}) doesn't match columns ({len(columns)}) for log_type '{log_type}'"
                    )
                row_data = list(table_data)
            else:
                raise ValueError(f"Data must be a list, tuple, or dict, got {type(table_data)}")
            

            self.wandb_table.add_data(*row_data)
            name = table_name or f"{log_type}_table"
            wandb.log({name: self.wandb_table})

        except Exception as e:
            print(f"Warning: Failed to log table to WandB: {e}")
    
    def _start_wandb_run(self, episode_id: Optional[Union[int, str]] = None):
        """Create a new WandB run"""
        if not (self.use_wandb and self.wandb_mode != "disabled"):
            return
        
        # Close previous run
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"Warning: Error finishing previous WandB run: {e}")
            self.wandb_run = None
        
        # Determine run name
        if episode_id is not None:
            run_name = f"episode-{episode_id}"
        else:
            run_name = self._base_run_name or self._timestamp
        
        try:            
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                group=self.wandb_group_name,
                name=run_name,
                config=self.wandb_config,
                mode=self.wandb_mode,
                reinit=True,
            )
            print(f"✓ WandB run started: {run_name}")
        except Exception as e:
            print(f"✗ Failed to start WandB run: {e}")
            self.wandb_run = None
    
    def start_episode(self, episode_id: Union[int, str]):
        """Start a new episode (creates a new run if using WandB)"""
        self._start_wandb_run(episode_id)
        print(f"Started episode: {episode_id}")
    
    def finish_episode(self):
        """Finish current episode"""
        self._finish_wandb_run_if_any()
    
    def _finish_wandb_run_if_any(self):
        """Finish the current WandB run"""
        if not (self.use_wandb and self.wandb_mode != "disabled"):
            return
        
        if self.wandb_run is not None:
            run_dir = getattr(self.wandb_run, "dir", None)
            try:
                self.wandb_run.finish()
                print("✓ WandB run finished")
            except Exception as e:
                print(f"Warning: Error finishing WandB run: {e}")
            finally:
                if self.wandb_mode == "offline" and run_dir:
                    print(f"WandB offline logs saved in: {run_dir}")
                    print(f"Sync with: wandb sync {run_dir}")
                self.wandb_run = None
    
    def flush_all(self):
        """Flush all cached data to files"""
        for log_type, cached_data in self.data_cache.items():
            if not cached_data:
                continue
            
            # Rewrite entire CSV file
            with open(self.csv_paths[log_type], 'a', newline='') as f:
                writer = csv.writer(f)
                # Write header
                # writer.writerow(self.log_types[log_type])
                # Write all data
                writer.writerows(cached_data)
                self.data_cache[log_type].clear()
        
        print(f"✓ All data flushed to {self.exp_log_dir}")
    
    def get_data(self, log_type: str) -> List[List[Any]]:
        """Get all cached data for a specific log type"""
        if log_type not in self.log_types:
            raise ValueError(f"Unknown log_type '{log_type}'")
        return self.data_cache[log_type].copy()
    
    def get_data_as_dict(self, log_type: str) -> List[Dict[str, Any]]:
        """Convert cached data to list of dictionaries format"""
        if log_type not in self.log_types:
            raise ValueError(f"Unknown log_type '{log_type}'")
        
        columns = self.log_types[log_type]
        return [
            {col: val for col, val in zip(columns, row)}
            for row in self.data_cache[log_type]
        ]
    
    def summary(self):
        """Print log summary"""
        print(f"\n{'='*60}")
        print(f"Experiment Summary: {self.experiment_name}")
        print(f"{'='*60}")
        for log_type, cached_data in self.data_cache.items():
            print(f"{log_type}: {len(cached_data)} entries")
        print(f"Log directory: {self.exp_log_dir}")
        print(f"{'='*60}\n")
    
    def close(self):
        """Close logger and save all data"""
        if not self.auto_flush:
            self.flush_all()
        self._finish_wandb_run_if_any()
        self.summary()
        print(f"✓ Logger closed. All logs saved to: {self.exp_log_dir}")
    
    def __enter__(self):
        """Support context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup on exit"""
        self.close()
        return False


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Define log types and columns
    log_config = {
        "training": ["epoch", "loss", "accuracy", "lr"],
        "validation": ["epoch", "val_loss", "val_accuracy"],
        "system": ["timestamp", "cpu_usage", "memory_usage", "gpu_temp"],
    }
    
    # Method 1: Use context manager (recommended)
    with ExperimentLogger(
        log_types=log_config,
        experiment_name="my_experiment",
        use_wandb=False,
        auto_flush=True
    ) as logger:
        # Log using list format
        logger.log("training", [1, 0.5, 0.85, 0.001])
        logger.log("training", [2, 0.3, 0.90, 0.001])
        
        # Log using dict format (more clear)
        logger.log("validation", {
            "epoch": 1,
            "val_loss": 0.6,
            "val_accuracy": 0.82
        })
        
        logger.log("system", {
            "timestamp": "2024-01-01 12:00:00",
            "cpu_usage": 45.2,
            "memory_usage": 8.5,
            "gpu_temp": 65
        })
    
    # Method 2: Manual management
    logger = ExperimentLogger(
        log_types=log_config,
        use_wandb=True,
        wandb_mode="offline",
        wandb_project="my_project"
    )
    
    # Log multiple episodes
    for episode in range(3):
        logger.start_episode(episode)
        
        for step in range(10):
            logger.log("training", {
                "epoch": step,
                "loss": 1.0 / (step + 1),
                "accuracy": 0.5 + 0.05 * step,
                "lr": 0.001
            })
        
        logger.finish_episode()
    
    logger.close()