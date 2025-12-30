import os
import argparse
import sys
import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def process_seed(engine, seed, turns, profile_id=None):
    seed_id = seed['seed_id']
    content = seed['content']
    try:
        trajectory = engine.run_session(seed_id, content, turns=turns, profile_id=profile_id)
        return seed_id, True, len(trajectory.conversation) // 2
    except Exception as e:
        return seed_id, False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Multi-turn dialogue simulation for anthropomorphism probing.")
    parser.add_argument("--csv_input", type=str, default="seed.csv", help="Path to input seed CSV.")
    parser.add_argument("--seeds_json", type=str, default="data/seeds.json", help="Path to intermediate seeds JSON.")
    parser.add_argument("--output_dir", type=str, default="data/trajectories", help="Directory to save trajectories.")
    
    # Model Selection
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini-0718", help="Model name for User model.")
    parser.add_argument("--user_channel", type=str, default="idealab", help="Channel for User (dashscope, whale, etc.).")
    
    parser.add_argument("--assistant_model", type=str, default="Oyster_7B_dpo", help="Model name for Assistant model.")
    parser.add_argument("--assistant_channel", type=str, default="whale", help="Channel for Assistant (dashscope, whale, etc.).")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of seeds to process.")
    parser.add_argument("--env_file", type=str, default=".env", help="Path to .env file.")
    parser.add_argument("--turns", type=int, default=3, help="Total number of dialogue rounds.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel sessions.")
    parser.add_argument("--profile_id", type=int, default=None, help="Specific profile ID to use for all sessions.")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(args.env_file)

    # 0. Generate Run ID and Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.user_model.split('/')[-1]}_vs_{args.assistant_model.split('/')[-1]}_{timestamp}"
    actual_output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(actual_output_dir, exist_ok=True)

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
        prober_client = LLMClient(model_name=args.user_model, channel=args.user_channel)
        mut_client = LLMClient(model_name=args.assistant_model, channel=args.assistant_channel)
    except ValueError as e:
        print(f"Error initializing models: {e}")
        return

    # 3. Initialize Engine
    engine = SimulationEngine(prober=prober_client, mut=mut_client, output_dir=actual_output_dir)

    # 4. Run Simulation in Parallel
    print(f"Starting simulation: User={args.user_model}, Assistant={args.assistant_model}, Parallel Workers={args.max_workers}")
    print(f"Output directory: {actual_output_dir}")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_seed, engine, seed, args.turns, args.profile_id): seed for seed in seeds}
        
        for future in tqdm(as_completed(futures), total=len(seeds), desc="Simulating"):
            seed_id, success, result = future.result()
            if success:
                tqdm.write(f"✓ {seed_id}: Completed {result} rounds.")
            else:
                tqdm.write(f"✗ {seed_id}: Failed: {result}")

if __name__ == "__main__":
    main()

