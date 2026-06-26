import json
import time
from pathlib import Path
from typing import Dict, Any
from inference import OllamaInference
from config import MODELS, BENCHMARK_PROMPTS

RESULTS_FILE = Path("benchmark_results.json")

def run_benchmark() -> Dict[str, Any]:
    """
    Benchmark all models on standardized prompts.
    Returns dict with throughput and latency metrics for each model.
    """
    inference = OllamaInference()
    results = {}

    for model_name in MODELS.keys():
        print(f"\nBenchmarking {model_name}...")
        model_results = {
            "times": [],
            "tokens": [],
            "throughputs": []
        }

        for i, prompt in enumerate(BENCHMARK_PROMPTS, 1):
            print(f"  Prompt {i}/{len(BENCHMARK_PROMPTS)}", end=" ", flush=True)

            try:
                start = time.time()
                response = inference.generate(
                    model=model_name,
                    prompt=prompt,
                    temperature=0.7,
                    num_predict=128
                )
                elapsed = time.time() - start

                tokens = response["tokens_generated"]
                throughput = tokens / elapsed if elapsed > 0 else 0

                model_results["times"].append(elapsed)
                model_results["tokens"].append(tokens)
                model_results["throughputs"].append(throughput)

                print(f"✓ ({elapsed:.2f}s, {throughput:.1f} tok/s)")

            except Exception as e:
                print(f"✗ Error: {e}")

        # Aggregate results
        if model_results["times"]:
            results[model_name] = {
                "avg_time": sum(model_results["times"]) / len(model_results["times"]),
                "min_time": min(model_results["times"]),
                "max_time": max(model_results["times"]),
                "avg_throughput": sum(model_results["throughputs"]) / len(model_results["throughputs"]),
                "min_throughput": min(model_results["throughputs"]),
                "max_throughput": max(model_results["throughputs"]),
                "total_tokens": sum(model_results["tokens"]),
                "runs": len(model_results["times"])
            }

    # Save results
    save_benchmark_results(results)
    return results

def save_benchmark_results(results: Dict[str, Any]) -> None:
    """Save benchmark results to file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

def load_benchmark_results() -> Dict[str, Any]:
    """Load benchmark results from file."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}

def print_benchmark_summary(results: Dict[str, Any]) -> None:
    """Print a formatted benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"  Avg Time:         {metrics['avg_time']:.2f}s")
        print(f"  Avg Throughput:   {metrics['avg_throughput']:.2f} tok/s")
        print(f"  Total Tokens:     {metrics['total_tokens']}")
        print(f"  Runs:             {metrics['runs']}")

    # Quality vs Speed comparison
    print("\n" + "-"*60)
    print("QUALITY vs SPEED TRADE-OFFS")
    print("-"*60)

    sorted_by_speed = sorted(
        results.items(),
        key=lambda x: x[1]["avg_throughput"],
        reverse=True
    )

    for i, (model_name, metrics) in enumerate(sorted_by_speed, 1):
        model_cfg = MODELS[model_name]
        print(f"\n{i}. {model_name}")
        print(f"   Speed Rank: #{i}")
        print(f"   Quality: {model_cfg['quality']}/10")
        print(f"   Throughput: {metrics['avg_throughput']:.2f} tok/s")
        print(f"   Best for: {model_cfg['use_case']}")
