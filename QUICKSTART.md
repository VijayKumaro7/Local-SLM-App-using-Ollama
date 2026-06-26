# Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

- **Ollama** installed: [download here](https://ollama.ai)
- **Python 3.8+**
- At least **2GB free VRAM** (more is better)

## Step 1: Start Ollama

```bash
ollama serve
```

(Leave this running in a separate terminal)

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Download Models (Optional but Recommended)

```bash
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull llama2:13b
```

Or run the setup script:
```bash
python setup.py
```

## Step 4: Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Step 5: Try It Out

### Option A: Generate Text
1. Go to **Generate** tab
2. Select a model
3. Enter a prompt
4. Click "Generate"
5. Watch real-time performance metrics

### Option B: Run Benchmark
1. Go to **Benchmark** tab
2. Click "Start Benchmark"
3. Wait 5-10 minutes
4. See which model is fastest/best

### Option C: View Comparison
1. Go to **Comparison** tab
2. See side-by-side metrics
3. Understand quality vs speed tradeoffs

## 🚨 Troubleshooting

### "Connection refused" error
→ Make sure Ollama is running (`ollama serve`)

### "Model not found"
→ Pull the model first:
```bash
ollama pull llama2:7b
```

### Very slow responses (< 1 tok/s)
→ You're running on CPU. For better speed:
- Install a GPU driver (NVIDIA/AMD)
- Or stick to 7B models on CPU
- Or reduce max_tokens in the app

### Out of memory errors
→ Switch to a smaller model (Llama 2 7B instead of 13B)

## 📚 Learn More

- See [README.md](README.md) for full documentation
- Ollama docs: https://github.com/jmorganca/ollama
- Streamlit docs: https://docs.streamlit.io

---

**That's it!** You're now running language models entirely offline on your machine. 🎉
