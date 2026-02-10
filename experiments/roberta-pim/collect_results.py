import os
import json
import argparse

# Define task order and corresponding result filenames and metric keys
# Order as per image: MNLI, QNLI, RTE, SST, MRPC, CoLA, QQP, STSB
TASKS = [
    {"name": "MNLI", "dir": "mnli", "metric": "accuracy"},
    {"name": "QNLI", "dir": "qnli", "metric": "accuracy"},
    {"name": "RTE",  "dir": "rte",  "metric": "accuracy"},
    {"name": "SST",  "dir": "sst2", "metric": "accuracy"},
    {"name": "MRPC", "dir": "mrpc", "metric": "accuracy"},
    {"name": "CoLA", "dir": "cola", "metric": "matthews_correlation"},
    {"name": "QQP",  "dir": "qqp",  "metric": "accuracy"},
    {"name": "STSB", "dir": "stsb", "metric": ["pearson", "spearmanr"]},
]

DEFAULT_ROOT_DIR = "experiments/roberta-pim/eval_results"

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_task_value(task, data):
    metric = task["metric"]
    if isinstance(metric, list):
        metric_vals = [safe_float(data.get(m, 0)) for m in metric]
        return sum(metric_vals) / len(metric_vals) if metric_vals else 0.0
    return safe_float(data.get(metric, 0))


def resolve_result_path(task_dir, result_file="all_results.json"):
    # Pattern 1: <task>/all_results.json
    direct_path = os.path.join(task_dir, result_file)
    if os.path.exists(direct_path):
        return direct_path

    # Pattern 2: <task>/<any_subdir>/all_results.json (e.g., lr_2e-5)
    if os.path.isdir(task_dir):
        for entry in sorted(os.listdir(task_dir)):
            candidate = os.path.join(task_dir, entry, result_file)
            if os.path.exists(candidate):
                return candidate

    return None


def collect(
    target_subfolder,
    root_dir,
    result_file="all_results.json",
    output_csv=None,
):
    base_dir = os.path.join(root_dir, target_subfolder)

    if not os.path.exists(base_dir):
        print(f"Error: Folder not found: {base_dir}")
        return

    print(f"\n>>> Analyzing directory: {base_dir}")
    
    header = []
    values = []
    
    for task in TASKS:
        task_dir = os.path.join(base_dir, task["dir"])
        path = resolve_result_path(task_dir, result_file=result_file)
        header.append(task["name"])
        
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    val = get_task_value(task, data)
                    values.append(f"{val:.4f}")
            except (json.JSONDecodeError, OSError):
                values.append("Error")
        else:
            values.append("N/A")

    # Calculate Avg
    valid_vals = []
    for v in values:
        try:
            valid_vals.append(float(v))
        except ValueError:
            continue
            
    avg = sum(valid_vals) / len(valid_vals) if valid_vals else 0
    header.append("Avg")
    values.append(f"{avg:.4f}")

    # Print formatted table
    col_width = 10
    print("-" * (col_width * len(header)))
    print("".join(word.ljust(col_width) for word in header))
    print("".join(word.ljust(col_width) for word in values))
    print("-" * (col_width * len(header)))

    # Save to CSV
    safe_target_name = os.path.basename(os.path.normpath(str(target_subfolder))) or "results"
    csv_path = output_csv or f"summary_{safe_target_name}.csv"
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(values) + "\n")
    print(f"\nResults exported to CSV: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect results with customizable root directory.")
    parser.add_argument("target", nargs="?", help="Target folder (e.g., sram, pcm).")
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT_DIR,
        help="Root directory containing target folders (default: experiments/roberta-pim/eval_results).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path. Default: summary_<target>.csv",
    )
    
    args = parser.parse_args()

    root_dir = args.root
    
    try:
        available_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except:
        available_folders = []
    
    target = args.target
    if not target:
        if available_folders:
            print(f"Available folders in {root_dir}: {', '.join(available_folders)}")
            target = input(f"Please enter the folder name to analyze (default sram): ").strip() or "sram"
        else:
            target = "sram"
    
    collect(
        target,
        root_dir=root_dir,
        output_csv=args.output,
    )
