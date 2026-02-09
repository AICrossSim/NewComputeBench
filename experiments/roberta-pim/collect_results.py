import os
import json
import sys

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

ROOT_EVAL_DIR = "experiments/roberta-pim/eval_results"

def collect(target_subfolder):
    base_dir = os.path.join(ROOT_EVAL_DIR, target_subfolder)
    if not os.path.exists(base_dir):
        print(f"Error: Folder not found: {base_dir}")
        return

    print(f"\n>>> Analyzing directory: {base_dir}")
    
    header = []
    values = []
    
    for task in TASKS:
        path = os.path.join(base_dir, task["dir"], "all_results.json")
        header.append(task["name"])
        
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if task["name"] == "STSB":
                        val = (data.get("pearson", 0) + data.get("spearmanr", 0)) / 2
                    else:
                        val = data.get(task["metric"], 0)
                    values.append(f"{val:.4f}")
            except:
                values.append("Error")
        else:
            values.append("N/A")

    # Calculate Avg
    valid_vals = []
    for v in values:
        try:
            valid_vals.append(float(v))
        except:
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
    csv_path = f"summary_{target_subfolder}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(values) + "\n")
    print(f"\nResults exported to CSV: {csv_path}")

if __name__ == "__main__":
    # Get available subfolders in eval_results (sram, pcm, reram, etc.)
    try:
        available_folders = [d for d in os.listdir(ROOT_EVAL_DIR) if os.path.isdir(os.path.join(ROOT_EVAL_DIR, d))]
    except:
        available_folders = []
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        if available_folders:
            print(f"Available folders: {', '.join(available_folders)}")
            target = input("Please enter the folder name to analyze (default sram): ").strip() or "sram"
        else:
            target = "sram"
    
    collect(target)
