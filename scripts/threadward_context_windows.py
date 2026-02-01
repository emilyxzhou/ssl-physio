"""
Threadward runner for Context Windows MAML experiment.

Variables:
- embedding_model: s4, mamba
- masking_ratio: masking_10, masking_30, masking_50, masking_70
- input_days: 3, 5, 7
- output_days: 1, 2, 3, 4, 5, 6, 7
- prediction_model: cnn, nn
- seed: 0, 1, 2, 3, 4

Total: 2 × 4 × 3 × 7 × 2 × 5 = 1,680 tasks
"""

import argparse
import os
import sys

# Add src folder to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import threadward

from context_windows.context_windows import run_context_windows


class ContextWindowsRunner(threadward.Threadward):
    def __init__(self, debug=False, results_folder="../context_windows_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        self.set_constraints(
            SUCCESS_CONDITION="NO_ERROR_AND_VERIFY",
            OUTPUT_MODE="LOG_FILE_ONLY",
            NUM_WORKERS=20,  # MAML is more GPU-intensive
            NUM_GPUS_PER_WORKER=0.1,
            NUM_CPUS_PER_WORKER=-1,
            AVOID_GPUS=None,
            INCLUDE_GPUS=[1,3],
            FAILURE_HANDLING="PRINT_FAILURE_AND_CONTINUE",
            TASK_FOLDER_LOCATION="VARIABLE_SUBFOLDER",
            EXISTING_FOLDER_HANDLING="VERIFY",
            TASK_TIMEOUT=-1
        )
    
    def task_method(self, variables, task_folder, log_file):
        # Extract variables from threadward
        embedding_model = variables["embedding_model"]
        masking_ratio = variables["masking_ratio"]
        days_config = variables["days_config"]
        prediction_model = variables["prediction_model"]
        seed = variables["seed"]
        
        # Parse days_config (format: "in_03_out_05")
        parts = days_config.split("_")
        input_days = int(parts[1])
        output_days = int(parts[3])
        
        # Run the context windows experiment
        run_context_windows(
            embedding_model=embedding_model,
            masking_ratio=masking_ratio,
            input_days=input_days,
            output_days=output_days,
            prediction_model=prediction_model,
            seed=seed,
            output_folder=task_folder,
            # MAML config (can be adjusted)
            inner_lr=0.001,  # Reduced from 0.01 to prevent gradient explosion
            outer_lr=0.001,
            inner_steps=5,
            meta_epochs=20,
            inner_steps_test=10,
            # Data config
            min_days_per_subject=30,
            num_support_subjects=32,
            samples_per_subject=5,
            query_stride=3,
            verbose=False
        )
    
    def verify_task_success(self, variables, task_folder, log_file):
        return os.path.exists(os.path.join(task_folder, "results.json"))
    
    def setup_variable_set(self, variable_set):
        # 1. Embedding model
        embedding_models = ["s4", "mamba"]
        variable_set.add_variable("embedding_model",
            values=embedding_models,
            nicknames=embedding_models)
        
        # 2. Masking ratio
        masking_ratios = ["masking_10", "masking_30", "masking_50", "masking_70"]
        variable_set.add_variable("masking_ratio",
            values=masking_ratios,
            nicknames=masking_ratios)
        
        # 3. Days in/out combinations
        input_days_options = [3, 5, 7]
        output_days_options = [1, 2, 3, 4, 5, 6, 7]
        
        days_values = []
        days_nicknames = []
        for inp in input_days_options:
            for out in output_days_options:
                combo = f"in_{inp:02d}_out_{out:02d}"
                days_values.append(combo)
                days_nicknames.append(combo)
        
        variable_set.add_variable("days_config",
            values=days_values,
            nicknames=days_nicknames)
        
        # 4. Prediction model
        prediction_models = ["cnn", "nn"]
        variable_set.add_variable("prediction_model",
            values=prediction_models,
            nicknames=prediction_models)
        
        # 5. Seed
        seeds = [0, 1, 2, 3, 4]
        variable_set.add_variable("seed",
            values=seeds,
            nicknames=[f"seed_{s}" for s in seeds])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Context Windows MAML via threadward')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--results-folder', default='../context_windows_results',
                       help='Name of the results folder')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ContextWindowsRunner(debug=args.debug, results_folder=args.results_folder)
    runner.run()

