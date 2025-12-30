import os
import argparse
import json
import csv
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.schemas import Trajectory
from src.models import LLMClient
from src.analyzer import BehaviorAnalyzer

def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

def run_single_analysis(analyzer: BehaviorAnalyzer, task: Dict[str, Any]):
    """
    Runs a single classifier check for a specific turn in a trajectory.
    """
    try:
        res = analyzer.analyze_chunk(task["classifier_name"], task["chunk"])
        return {
            "seed_id": task["seed_id"],
            "turn_index": task["turn_index"],
            "classifier_name": task["classifier_name"],
            "is_detected": res["is_detected"],
            "confidence": res["confidence"],
            "mut_response_snippet": task["snippet"]
        }
    except Exception as e:
        return {
            "seed_id": task["seed_id"],
            "turn_index": task["turn_index"],
            "classifier_name": task["classifier_name"],
            "is_detected": False,
            "confidence": 0,
            "mut_response_snippet": f"ERROR: {str(e)[:50]}"
        }

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectories for over-anthropomorphism.")
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectories", help="Directory containing trajectory JSONs.")
    parser.add_argument("--output_file", type=str, default="data/analysis_results/summary.csv", help="Output summary CSV path.")
    parser.add_argument("--classifiers", type=str, default="attribute_emotions,emotional_bond", help="Comma-separated or 'all'.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model to use as judge.")
    parser.add_argument("--judge_channel", type=str, default="idealab", help="Channel for judge model.")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of parallel API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trajectory files to process.")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file.")

    args = parser.parse_args()
    load_dotenv(args.env_file)

    # 1. Setup paths and auto-detect latest subdirectory
    if args.trajectory_dir == "data/trajectories":
        subdirs = [os.path.join(args.trajectory_dir, d) for d in os.listdir(args.trajectory_dir) 
                   if os.path.isdir(os.path.join(args.trajectory_dir, d))]
        if subdirs:
            # Sort by modification time to get the latest
            args.trajectory_dir = max(subdirs, key=os.path.getmtime)
            print(f"Auto-detected latest trajectory directory: {args.trajectory_dir}")
        else:
            print(f"No subdirectories found in {args.trajectory_dir}.")
            return

    # Use run_id for output filename if default is used
    if args.output_file == "data/analysis_results/summary.csv":
        run_id = os.path.basename(args.trajectory_dir)
        args.output_file = f"data/analysis_results/analysis_{run_id}.csv"

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 2. Initialize Analyzer
    try:
        judge_client = LLMClient(model_name=args.judge_model, channel=args.judge_channel)
        analyzer = BehaviorAnalyzer(judge_client)
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return

    # 3. Setup Classifiers
    if args.classifiers.lower() == "all":
        classifier_list = list(analyzer.definitions.keys())
    else:
        classifier_list = [c.strip() for c in args.classifiers.split(",")]
    
    # Check for invalid classifiers
    invalid = [c for c in classifier_list if c not in analyzer.definitions]
    if invalid:
        print(f"Warning: Classifiers not found in definitions: {invalid}")
        classifier_list = [c for c in classifier_list if c in analyzer.definitions]

    if not classifier_list:
        print("No valid classifiers specified.")
        return

    # 4. Collect files and prepare tasks
    files = [os.path.join(args.trajectory_dir, f) for f in os.listdir(args.trajectory_dir) if f.endswith(".json")]
    files.sort()
    
    if args.limit:
        files = files[:args.limit]
        print(f"Limited to {len(files)} files.")

    if not files:
        print(f"No trajectory files found in {args.trajectory_dir}.")
        return

    # Construct flat task list for better progress tracking
    all_tasks = []
    print(f"Preparing tasks for {len(files)} trajectories and {len(classifier_list)} classifiers...")
    for fp in files:
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            trajectory = Trajectory(**data)
            convo_dicts = [m.model_dump() for m in trajectory.conversation]
            
            target_indices = []
            if len(convo_dicts) > 3: target_indices.append(3)
            if len(convo_dicts) > 5: target_indices.append(5)
            
            for idx in target_indices:
                import emoclassifiers.chunking as chunking
                chunk = chunking.Chunk.from_simple_convo(convo_dicts, idx=idx, n_context=3)
                turn_index = (idx // 2) + 1
                
                for cls_name in classifier_list:
                    all_tasks.append({
                        "seed_id": trajectory.metadata.seed_id,
                        "turn_index": turn_index,
                        "classifier_name": cls_name,
                        "chunk": chunk,
                        "snippet": convo_dicts[idx]["content"][:100] + "..."
                    })
        except Exception as e:
            print(f"Skipping {fp} due to error: {e}")

    if not all_tasks:
        print("No analysis tasks were generated.")
        return

    print(f"Total analysis tasks: {len(all_tasks)}. Starting with {args.max_workers} workers.")

    # 5. Run Analysis with Granular Progress
    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_single_analysis, analyzer, task): task for task in all_tasks}
        
        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Detailed Analysis"):
            result = future.result()
            all_results.append(result)

    # 6. Save Results
    if all_results:
        keys = ["seed_id", "turn_index", "classifier_name", "is_detected", "confidence", "mut_response_snippet"]
        # Sort results by seed_id and turn_index
        all_results.sort(key=lambda x: (x["seed_id"], x["turn_index"]))
        
        with open(args.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nAnalysis complete. Results saved to {args.output_file}")
    else:
        print("\nNo analysis results generated.")

if __name__ == "__main__":
    main()
