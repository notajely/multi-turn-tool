import os
import argparse
import sys
from src.utils import csv_to_json
from src.models import LLMClient
from src.engine import SimulationEngine

def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

def main():
    parser = argparse.ArgumentParser(description="Multi-turn dialogue simulation for anthropomorphism probing.")
    parser.add_argument("--csv_input", type=str, default="seed.csv", help="Path to input seed CSV.")
    parser.add_argument("--seeds_json", type=str, default="data/seeds.json", help="Path to intermediate seeds JSON.")
    parser.add_argument("--output_dir", type=str, default="data/trajectories", help="Directory to save trajectories.")
    
    # Model Selection
    parser.add_argument("--prober_model", type=str, default="claude37_sonnet", help="Model name for Prober.")
    parser.add_argument("--prober_channel", type=str, default="idealab", help="Channel for Prober (dashscope, whale, etc.).")
    
    parser.add_argument("--mut_model", type=str, default="qwen2.5-72b-instruct", help="Model name for MUT.")
    parser.add_argument("--mut_channel", type=str, default="dashscope", help="Channel for MUT (dashscope, whale, etc.).")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds to process.")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file.")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    # 1. Convert CSV to JSON
    if not os.path.exists(args.csv_input):
        print(f"Error: CSV input file '{args.csv_input}' not found.")
        return

    print(f"Converting CSV {args.csv_input} to {args.seeds_json}...")
    seeds = csv_to_json(args.csv_input, args.seeds_json)

    if args.limit:
        seeds = seeds[:args.limit]
        print(f"Limited to {len(seeds)} seeds.")

    # 2. Initialize Models
    try:
        prober_client = LLMClient(model_name=args.prober_model, channel=args.prober_channel)
        mut_client = LLMClient(model_name=args.mut_model, channel=args.mut_channel)
    except ValueError as e:
        print(f"Error initializing models: {e}")
        return

    # 3. Initialize Engine
    engine = SimulationEngine(prober=prober_client, mut=mut_client, output_dir=args.output_dir)

    # 4. Run Simulation
    for seed in seeds:
        seed_id = seed['seed_id']
        content = seed['content']
        try:
            engine.run_session(seed_id, content)
            print(f"Successfully processed {seed_id}")
        except Exception as e:
            print(f"Failed to process {seed_id}: {e}")
            # Continue to next seed

if __name__ == "__main__":
    main()
