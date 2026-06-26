#!/usr/bin/env python
"""
Command-line benchmark runner for Local SLM.
Run without the Streamlit UI for automated benchmarking.

Usage:
    python benchmark_cli.py              # Run full benchmark
    python benchmark_cli.py --model llama2:7b  # Benchmark single model
    python benchmark_cli.py --quick     # Quick 5-prompt benchmark
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from inference import OllamaInference
from config import MODELS, BENCHMARK_PROMPTS

def format_results(results: dict) -> str:
    """Format results as a nice table."""
    lines = [
        "\n" + "="*70,
        "BENCHMARK RESULTS",
        "="*70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    # Sort by throughput (speed)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["avg_throughput"],
        reverse=True
    )

    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        model_cfg = MODELS[model_name]
        lines.append(f"{rank}. {model_name.upper()}")
        lines.append(f"   Parameters:       {model_cfg['params']}")
        lines.append(f"   Quality Rating:   {model_cfg['quality']}/10")
        lines.append(f"   Use Case:         {model_cfg['use_case']}")
        lines.append("")
        lines.append(f"   Speed (tok/s):    {metrics['avg_throughput']:.2f} avg | "
                     f"{metrics['min_throughput']:.2f} min | {metrics['max_throughput']:.2f} max")
        lines.append(f"   Latency (s):      {metrics['avg_time']:.2f} avg | "
                     f"{metrics['min_time']:.2f} min | {metrics['max_time']:.2f} max")
        lines.append(f"   Total Tokens:     {metrics['total_tokens']}")
        lines.append(f"   Runs:             {metrics['runs']}")
        lines.append("")

    # Summary
    lines.append("-"*70)
    lines.append("QUALITY vs SPEED SUMMARY")
    lines.append("-"*70)
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        quality = MODELS[model_name]["quality"]
        speed = metrics["avg_throughput"]
        efficiency = quality / (50 / speed)  # Quality per latency unit
        lines.append(f"{rank}. {model_name:15} | Speed: {speed:6.2f} tok/s | "
                     f"Quality: {quality}/10 | Efficiency: {efficiency:.2f}")

    lines.append("="*70 + "\n")
    return "\n".join(lines)

def run_model_benchmark(model_name: str, prompts: list, inference: OllamaInference) -> dict:
    """Benchmark a single model."""
    times = []
    tokens = []
    throughputs = []

    print(f"\n📊 Benchmarking {model_name}")
    print("-" * 50)

    for i, prompt in enumerate(prompts, 1):
        pct = (i / len(prompts)) * 100
        print(f"  [{i:2d}/{len(prompts)}] ({pct:5.1f}%)", end=" ", flush=True)

        try:
            start = time.time()
            response = inference.generate(
                model=model_name,
                prompt=prompt,
                temperature=0.7,
                num_predict=128
            )
            elapsed = time.time() - start

            token_count = response["tokens_generated"]
            throughput = token_count / elapsed if elapsed > 0 else 0

            times.append(elapsed)
            tokens.append(token_count)
            throughputs.append(throughput)

            status = "✓"
            print(f"{status} {elapsed:6.2f}s | {throughput:6.1f} tok/s")

        except Exception as e:
            print(f"✗ ERROR: {e}")

    if not times:
        return {}

    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "avg_throughput": sum(throughputs) / len(throughputs),
        "min_throughput": min(throughputs),
        "max_throughput": max(throughputs),
        "total_tokens": sum(tokens),
        "runs": len(times)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Local SLM models"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Benchmark specific model (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with 5 prompts instead of 13"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    inference = OllamaInference()

    # Check connection
    try:
        inference.list_models()
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running (ollama serve)")
        return False

    # Select prompts
    prompts = BENCHMARK_PROMPTS[:5] if args.quick else BENCHMARK_PROMPTS

    # Select models
    models_to_benchmark = (
        [args.model] if args.model
        else list(MODELS.keys())
    )

    # Run benchmarks
    results = {}
    for model in models_to_benchmark:
        result = run_model_benchmark(model, prompts, inference)
        if result:
            results[model] = result

    # Display results
    if results:
        print(format_results(results))

        # Save to file
        if not args.no_save:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to {output_path}\n")

        return True
    else:
        print("❌ No benchmark results")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
