"""Configuration for models, benchmarks, and Ollama connection."""

import os

# Host of the Ollama server. Defaults to local, but can be overridden via the
# OLLAMA_HOST environment variable / Streamlit secret so the app can point at a
# remote Ollama instance when deployed online.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Model definitions with metadata.
# The lightweight models are listed first so they are the default selection and
# easy to pull/test (a few hundred MB to ~1GB) without the larger 7B/13B models.
MODELS = {
    "qwen2.5:0.5b": {
        "name": "Qwen 2.5 0.5B",
        "params": "0.5B",
        "speed": "Very Fast (~80 tok/s)",
        "quality": 5,
        "description": "Tiny model (~400MB), great for quick local testing on CPU",
        "use_case": "Smoke tests, low-resource devices, fast prototyping",
        "pros": ["Tiny download", "Runs well on CPU", "Very fast"],
        "cons": ["Lower quality", "Limited reasoning"],
    },
    "llama3.2:1b": {
        "name": "Llama 3.2 1B",
        "params": "1B",
        "speed": "Very Fast (~60 tok/s)",
        "quality": 6,
        "description": "Small model (~1.3GB), good speed/quality balance for testing",
        "use_case": "Lightweight chat, quick demos, resource-constrained setups",
        "pros": ["Small download", "Fast on CPU", "Decent quality for size"],
        "cons": ["Less capable than 7B+ models"],
    },
    "llama2:7b": {
        "name": "Llama 2 7B",
        "params": "7B",
        "speed": "Fast (~30 tok/s)",
        "quality": 7,
        "description": "Balanced model, fastest inference, good for real-time applications",
        "use_case": "Real-time chat, content generation, responsive UX",
        "pros": ["Fastest inference", "Lower memory", "Good quality"],
        "cons": ["Lower reasoning capability", "Less nuanced responses"],
    },
    "mistral:7b": {
        "name": "Mistral 7B",
        "params": "7B",
        "speed": "Fast (~25 tok/s)",
        "quality": 8,
        "description": "Efficient 7B model with better quality than Llama 2 7B",
        "use_case": "Balanced speed/quality, code generation, complex tasks",
        "pros": ["Better quality than Llama 2 7B", "Efficient architecture", "Good for coding"],
        "cons": ["Slightly slower than Llama 2 7B"],
    },
    "llama2:13b": {
        "name": "Llama 2 13B",
        "params": "13B",
        "speed": "Moderate (~12 tok/s)",
        "quality": 9,
        "description": "Largest model, best reasoning and knowledge, slower inference",
        "use_case": "Complex reasoning, detailed analysis, high-quality content",
        "pros": ["Best quality", "Excellent reasoning", "Nuanced responses"],
        "cons": ["Slowest inference", "Higher memory requirement"],
    },
}

# Benchmark prompts for standardized evaluation
BENCHMARK_PROMPTS = [
    # Knowledge/reasoning
    "Explain how photosynthesis works in simple terms.",
    "What are the main differences between machine learning and deep learning?",
    "Describe the water cycle in 3-4 sentences.",

    # Coding
    "Write a Python function that checks if a number is prime.",
    "How would you implement a simple stack data structure in Python?",

    # Creative
    "Write a short poem about the ocean.",
    "Create a fictional story opening that hooks the reader.",

    # Question answering
    "Who was the first person to walk on the moon?",
    "What year did World War II end?",

    # Writing/summarization
    "Summarize the benefits of regular exercise.",
    "List 5 tips for better sleep hygiene.",

    # Problem solving
    "How would you break down a complex project into manageable tasks?",
    "What's a good approach to debugging code?",
]

# Constraints documentation
CONSTRAINTS = {
    "privacy": {
        "description": "All inference happens locally - no data sent to external APIs",
        "benefit": "Complete data privacy for sensitive information"
    },
    "latency": {
        "description": "Faster response times for non-streaming inference vs cloud APIs",
        "note": "Speed varies based on hardware (GPU/CPU, RAM, disk I/O)"
    },
    "cost": {
        "description": "No API call costs - just local hardware utilization",
        "consideration": "Requires upfront hardware investment and electricity costs"
    },
    "offline": {
        "description": "Works without internet connection after models are downloaded",
        "benefit": "Reliable performance, no network dependency"
    }
}
