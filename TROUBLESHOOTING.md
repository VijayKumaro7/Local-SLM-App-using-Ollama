# Troubleshooting Guide

Common issues and solutions.

## Installation & Setup

### ❌ "ollama: command not found"
**Problem**: Ollama not installed or not in PATH

**Solution**:
1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and follow platform-specific setup
3. Add to PATH if needed:
   - **macOS/Linux**: Usually automatic
   - **Windows**: Should be in PATH after install
4. Verify: `ollama --version`

### ❌ "Connection refused" when app starts
**Problem**: Ollama server not running

**Solution**:
```bash
# In a separate terminal, start Ollama
ollama serve
# You should see: "Listening on 127.0.0.1:11434"
```

### ❌ "ModuleNotFoundError: No module named 'streamlit'"
**Problem**: Python dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt
```

### ❌ "Python 3.7 or lower detected"
**Problem**: Python version too old

**Solution**:
1. Download Python 3.8+ from [python.org](https://python.org)
2. Update PATH to prefer new version
3. Verify: `python --version` (should be 3.8+)

## Running the App

### ❌ "No module named 'ollama'"
**Problem**: Ollama Python library not installed (different from Ollama server)

**Solution**:
```bash
pip install ollama
# Or reinstall all deps:
pip install -r requirements.txt
```

### ❌ "Connection timed out" during generation
**Problem**: Ollama server crashed or unresponsive

**Solution**:
1. Check if Ollama process is running (look for `ollama serve` window)
2. Restart Ollama: `ollama serve` in new terminal
3. Try again

### ❌ Streamlit shows "Port 8501 already in use"
**Problem**: Another Streamlit instance running

**Solution**:
```bash
# Option 1: Use different port
streamlit run app.py --server.port 8502

# Option 2: Kill existing process
# On Windows:
taskkill /IM python.exe /F

# On macOS/Linux:
pkill -f "streamlit run"
```

## Model Issues

### ❌ "Model not found" when generating
**Problem**: Selected model not downloaded

**Solution**:
```bash
# Download the model
ollama pull llama2:7b
# Or use setup script
python setup.py
```

### ❌ Generation is very slow (< 1 tok/s)
**Problem**: Running on CPU instead of GPU

**Solution**:
1. **Check GPU detection**:
   ```bash
   # For NVIDIA
   nvidia-smi
   
   # For AMD (Windows)
   dxdiag (check Display Devices)
   ```

2. **Install proper drivers**:
   - **NVIDIA**: Install CUDA toolkit and cuDNN
   - **AMD**: Install ROCm (Windows: Windows AMD Radeon Pro drivers)
   - **macOS**: Already optimized for Metal

3. **If stuck on CPU**:
   - Use smaller models (7B instead of 13B)
   - Accept slower speed or reduce `max_tokens`
   - Consider GPU hardware upgrade

### ❌ "CUDA out of memory" or similar
**Problem**: Model doesn't fit in VRAM

**Solution**:
1. Close other GPU applications (games, video editing, etc.)
2. Switch to smaller model:
   - Using 13B? Switch to 7B
   - Using 7B? Reduce `max_tokens` to 128
3. Check available VRAM:
   ```bash
   nvidia-smi -l 1  # Updates every second
   ```

### ❌ Model download fails ("No space left on device")
**Problem**: Disk full

**Solution**:
1. Check available space: `df -h` (macOS/Linux) or disk management (Windows)
2. Free up space (models are 4-13GB each)
3. Resume download:
   ```bash
   ollama pull llama2:7b
   ```

## Benchmark Issues

### ❌ Benchmark takes forever (> 30 minutes)
**Problem**: Running on slow hardware (CPU)

**Solution**:
1. **Cancel** current run (Ctrl+C)
2. **Use `--quick` flag**:
   ```bash
   python benchmark_cli.py --quick
   ```
3. **Benchmark one model at a time**:
   ```bash
   python benchmark_cli.py --model llama2:7b
   ```

### ❌ Benchmark shows unrealistic throughput (< 1 tok/s)
**Problem**: GPU not being used

**Solution**:
See "Generation is very slow" section above. Likely CPU-only mode.

### ❌ Benchmark results are inconsistent
**Problem**: Normal for CPU; disk I/O interference

**Solution**:
1. Close background apps
2. Run benchmark multiple times, take average
3. GPU results should be more stable

## Ollama Server Issues

### ❌ "bind: address already in use" error
**Problem**: Another Ollama instance running on port 11434

**Solution**:
```bash
# Option 1: Kill existing Ollama
# Windows:
taskkill /IM ollama.exe /F

# macOS:
pkill ollama

# Option 2: Use different port
OLLAMA_HOST=127.0.0.1:11435 ollama serve
# Then update config.py: OLLAMA_HOST = "http://localhost:11435"
```

### ❌ Ollama crashes or becomes unresponsive
**Problem**: Out of memory, GPU driver crash

**Solution**:
1. Restart Ollama: Close and re-run `ollama serve`
2. Check logs for details:
   - Windows: Look in C:\Users\<user>\.ollama\
   - macOS: `~/.ollama/`
   - Linux: `~/.ollama/`

3. Reduce memory usage:
   - Use smaller models (7B vs 13B)
   - Set `num_ctx` lower in requests

### ❌ High GPU memory usage stays after generation
**Problem**: Model stays in VRAM (by design for fast reuse)

**Solution**:
1. This is normal and desirable
2. To free VRAM, restart Ollama
3. Or just proceed to next inference (reuses cached model)

## Configuration Issues

### ❌ Changes to config.py not taking effect
**Problem**: Python module caching

**Solution**:
1. Restart Streamlit app: Stop and run `streamlit run app.py`
2. If using CLI: `python benchmark_cli.py` should pick up changes

### ❌ Environment variables not working
**Problem**: .env file not loaded

**Solution**:
1. Rename .env.example to .env:
   ```bash
   cp .env.example .env  # macOS/Linux
   copy .env.example .env  # Windows
   ```
2. Edit .env with your settings
3. Restart app to load

## Performance Optimization

### ⚡ Slow response times?
1. **Check hardware**:
   ```bash
   nvidia-smi  # GPU info
   ```

2. **Profile generation**:
   - Temperature 0.0 = fastest (most deterministic)
   - Lower num_predict = faster but shorter responses
   - Smaller models = faster

3. **Expected speeds**:
   - 7B on GPU: 20-40 tok/s
   - 7B on CPU: 2-8 tok/s
   - 13B on GPU: 10-20 tok/s

### 💾 High memory usage?
1. Close other applications
2. Use smaller models
3. Restart Ollama to clear cached models

## Help & Support

### If issues persist:
1. Check Ollama docs: https://github.com/jmorganca/ollama
2. Check Streamlit docs: https://docs.streamlit.io
3. Look at app logs for detailed error messages
4. Try setup verification:
   ```bash
   python setup.py
   ```

### Collecting debug info for support:
```bash
# System info
python --version
ollama --version

# Test Ollama
curl http://localhost:11434/api/tags

# Test models
ollama list

# Generate sample
ollama run llama2:7b "Hello"
```

---

**Still stuck?** Try the QUICKSTART.md from scratch on a fresh terminal.
