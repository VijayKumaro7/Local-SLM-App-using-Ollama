#!/usr/bin/env python
"""
Setup script for Local SLM Benchmark.
Checks prerequisites and pulls required models.
"""

import subprocess
import sys
import requests
from pathlib import Path
from inference import OllamaInference
from config import MODELS, OLLAMA_HOST

def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception as e:
        print(f"[ERROR] Ollama is not running: {e}")
        print(f"   Start Ollama with: ollama serve")
        return False

def check_python_version() -> bool:
    """Check Python version."""
    if sys.version_info >= (3, 8):
        print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} (required: 3.8+)")
        return True
    print(f"[ERROR] Python {sys.version_info.major}.{sys.version_info.minor} (required: 3.8+)")
    return False

def check_dependencies() -> bool:
    """Check if required Python packages are installed."""
    required = ["streamlit", "ollama", "requests", "pandas"]
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            missing.append(package)
            print(f"[ERROR] {package} (install with: pip install {package})")

    if missing:
        print("\n[INSTALL] Missing packages:")
        print(f"   pip install -r requirements.txt")
        return False
    return True

def check_models(inference: OllamaInference) -> list:
    """Check which models are available locally."""
    available = []
    required = list(MODELS.keys())

    try:
        local_models = inference.list_models()
        available = [m for m in required if any(m in lm for lm in local_models)]

        print("\nModels:")
        for model in required:
            if model in available:
                print(f"   [OK] {model}")
            else:
                print(f"   [MISSING] {model} (not downloaded)")

        missing = [m for m in required if m not in available]
        return missing

    except Exception as e:
        print(f"[ERROR] Could not check models: {e}")
        return required

def pull_models(inference: OllamaInference, models: list) -> bool:
    """Download missing models."""
    if not models:
        print("[OK] All models are available locally")
        return True

    print(f"\nPulling {len(models)} model(s)...")
    print("(This may take 10-30 minutes depending on your internet speed)\n")

    all_success = True
    for model in models:
        print(f"   Pulling {model}...", end=" ", flush=True)
        try:
            if inference.pull_model(model):
                print("[OK]")
            else:
                print("[ERROR]")
                all_success = False
        except Exception as e:
            print(f"[ERROR] ({e})")
            all_success = False

    return all_success

def main():
    print("Local SLM Setup\n")
    print("=" * 50)

    # Check Python version
    print("\n[1] Python Version")
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher")
        return False

    # Check dependencies
    print("\n[2] Python Dependencies")
    if not check_dependencies():
        print("\nPlease install missing dependencies")
        return False

    # Check Ollama
    print("\n[3] Ollama Server")
    if not check_ollama():
        return False
    print("[OK] Ollama is running")

    # Check models
    print("\n[4] Models")
    inference = OllamaInference()
    missing = check_models(inference)

    if missing:
        pull = input("\nDownload missing models? (y/n) ").lower().strip() == "y"
        if pull:
            if not pull_models(inference, missing):
                print("\n[WARNING] Some models failed to download")
        else:
            print("\nNote: You can pull models later with 'ollama pull <model-name>'")

    # Setup .env
    print("\n[5] Configuration")
    env_file = Path(".env")
    if not env_file.exists():
        print("   Creating .env file...")
        with open(".env.example") as f:
            content = f.read()
        with open(".env", "w") as f:
            f.write(content)
        print("   [OK] .env created (customize as needed)")
    else:
        print("   [OK] .env already exists")

    print("\n" + "=" * 50)
    print("[OK] Setup complete!\n")
    print("Next steps:")
    print("  1. Run the app: streamlit run app.py")
    print("  2. Open http://localhost:8501 in your browser")
    print("  3. Try generating text or run a benchmark\n")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
