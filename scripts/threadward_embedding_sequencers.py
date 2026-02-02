import argparse
import os
import sys

# Add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import threadward

from embedding_sequencers.embedding_sequencer import run_embedding_sequencer

class EmbeddingSequencersRunner(threadward.Threadward):
    def __init__(self, debug=False, results_folder="../embedding_sequencers_results"):
        super().__init__(debug=debug, results_folder=results_folder)
        self.set_constraints(
            SUCCESS_CONDITION="NO_ERROR_AND_VERIFY",
            OUTPUT_MODE="LOG_FILE_ONLY",
            NUM_WORKERS=40,
            NUM_GPUS_PER_WORKER=0.1,
            NUM_CPUS_PER_WORKER=-1,  # CPU cores per worker (-1 for no limit, Linux only)
            AVOID_GPUS=None,
            INCLUDE_GPUS=None,
            FAILURE_HANDLING="PRINT_FAILURE_AND_CONTINUE",
            TASK_FOLDER_LOCATION="VARIABLE_SUBFOLDER",
            EXISTING_FOLDER_HANDLING="VERIFY",
            TASK_TIMEOUT=-1  # Timeout in seconds (-1 for no timeout)
        )
    
    def task_method(self, variables, task_folder, log_file):
        # Extract variables from threadward
        masking_model = variables["masking_model"]
        days_in_out = variables["days_in_out"]
        prediction_model = variables["prediction_model"]
        
        # Parse days_in_out (format: "given_03_predict_07")
        parts = days_in_out.split("_")
        days_given = int(parts[1])
        days_predicted = int(parts[3])
        
        # Call the embedding sequencer
        run_embedding_sequencer(
            masking_model=masking_model,
            days_given=days_given,
            days_predicted=days_predicted,
            prediction_model=prediction_model,
            output_folder=task_folder,
            min_days_per_subject=30
        )
    
    def verify_task_success(self, variables, task_folder, log_file):
        return os.path.exists(os.path.join(task_folder, "results.json"))
    
    def setup_variable_set(self, variable_set):
        # 1. Masking model (first level of folder hierarchy)
        masking_models = ["masking_10", "masking_30", "masking_50"]
        variable_set.add_variable("masking_model",
            values=masking_models,
            nicknames=masking_models)
        
        # 2. Days in/out combinations (second level of folder hierarchy)
        days_given_options = [3, 5, 7]
        days_predicted_options = [1, 5, 7, 14]
        
        days_in_out_values = []
        days_in_out_nicknames = []
        for dg in days_given_options:
            for dp in days_predicted_options:
                combo = f"given_{dg:02d}_predict_{dp:02d}"
                days_in_out_values.append(combo)
                days_in_out_nicknames.append(combo)
        
        variable_set.add_variable("days_in_out",
            values=days_in_out_values,
            nicknames=days_in_out_nicknames)
        
        # 3. Prediction model (third level of folder hierarchy)
        prediction_models = ["cnn", "nn"]
        variable_set.add_variable("prediction_model",
            values=prediction_models,
            nicknames=prediction_models)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run embedding sequencers via threadward')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--results-folder', default='../embedding_sequencers_results',
                       help='Name of the results folder (default: embedding_sequencers_results)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    runner = EmbeddingSequencersRunner(debug=args.debug, results_folder=args.results_folder)
    runner.run()
