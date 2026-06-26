"""Configuration for models, benchmarks, and Ollama connection."""

OLLAMA_HOST = "http://localhost:11434"

# Model definitions with metadata
MODELS = {
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
