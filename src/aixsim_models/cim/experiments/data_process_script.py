import json
import os
from pathlib import Path

def load_results_from_cache(result_path):
    """
    Load all results from .cache folder and organize them according to the table format.
    Returns a structured dictionary with results organized by method and task.
    """
    
    # Metric mapping for each task (based on GLUE benchmark standards)
    metric_mapping = {
        "mnli": "eval_accuracy",
        "qnli": "eval_accuracy", 
        "rte": "eval_accuracy",
        "sst2": "eval_accuracy",
        "mrpc": "eval_accuracy",
        "cola": "eval_matthews_correlation",
        "qqp": "eval_accuracy",
        "stsb": "eval_pearson"
    }
    
    results = {}
    result_path = Path(result_path)
    
    # Load original results
    for item, value in metric_mapping.items():
        target_path = result_path / item / "all_results.json"
        if target_path.exists():    
            try:
                with open(target_path, 'r') as f:
                    data = json.load(f)
                results[item] = data[value]
            except Exception as e:
                print(f"Error loading {target_path}: {e}")
    return results

def save_organized_results(results, output_file="organized_results.json"):
    """
    Save the organized results to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Load and organize results
    print("Loading results from .cache folder...")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True, help='Path to the results directory')
    args = parser.parse_args()
    
    result_path = args.result_path

    results = load_results_from_cache(result_path)
    task_order = ["mnli", "qnli", "rte", "sst2", "mrpc", "cola", "qqp", "stsb"]
    rounded_results = {task: round(results[task], 4) for task in task_order}
    result_str = " & ".join([str(rounded_results[task]) for task in task_order])
    print(task_order)
    print(result_str)
    rounded_results["overleaf_format"] = result_str
    save_organized_results(rounded_results, result_path + "packed_results.json")
    print("Detailed results with metadata saved to packed_results.json")
