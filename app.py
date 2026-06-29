import streamlit as st
import time
import json
from datetime import datetime
from inference import OllamaInference, OllamaError
from benchmark import run_benchmark, load_benchmark_results
from config import MODELS, BENCHMARK_PROMPTS

st.set_page_config(
    page_title="Local SLM Benchmark",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Local SLM Benchmark with Ollama")
st.markdown("Compare inference speed and quality across 3 models running locally.")

# Initialize session state
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = load_benchmark_results()
if "inference_history" not in st.session_state:
    st.session_state.inference_history = []

# Shared Ollama client + availability detection
inference = OllamaInference()
ollama_up = inference.is_available()
installed_models = inference.list_models() if ollama_up else []


def model_installed(name: str) -> bool:
    """True if a configured model is present locally (tolerant of :latest)."""
    base = name.split(":")[0]
    return name in installed_models or any(
        m.split(":")[0] == base for m in installed_models
    )


# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Generate", "Benchmark", "Comparison"])

# Connection status
if ollama_up:
    st.sidebar.success("🟢 Ollama connected")
else:
    st.sidebar.error("🔴 Ollama not reachable — start it with `ollama serve`")

# Model selector — label each model with its install status
st.sidebar.markdown("### Model Settings")


def model_label(name: str) -> str:
    if not ollama_up:
        return name
    return f"{name} {'✅' if model_installed(name) else '⬇️ not pulled'}"


selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(MODELS.keys()),
    format_func=model_label,
    help="Choose which model to use for inference"
)

model_info = MODELS[selected_model]
st.sidebar.info(
    f"**{selected_model}**\n\n"
    f"Params: {model_info['params']}\n"
    f"Est. Speed: {model_info['speed']}\n"
    f"Est. Quality: {model_info['quality']}"
)

# Offer to pull the selected model if it isn't installed
if ollama_up and not model_installed(selected_model):
    st.sidebar.warning(f"`{selected_model}` is not pulled yet.")
    if st.sidebar.button(f"⬇️ Pull {selected_model}", use_container_width=True):
        with st.spinner(f"Pulling {selected_model} (this can take a while)..."):
            if inference.pull_model(selected_model):
                st.sidebar.success(f"Pulled {selected_model}!")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to pull {selected_model}.")

if page == "Generate":
    st.header("🎨 Text Generation")

    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            height=120,
            placeholder="e.g., 'Explain quantum computing in simple terms'"
        )

    with col2:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 50, 2000, 256)
        run_button = st.button("Generate", type="primary", use_container_width=True)

    if run_button and prompt:
        st.divider()

        # Display results — stream tokens as they arrive
        st.markdown("### Generated Output")
        stats = {}
        try:
            start_time = time.time()
            output_text = st.write_stream(
                inference.generate_stream(
                    model=selected_model,
                    prompt=prompt,
                    stats=stats,
                    temperature=temperature,
                    num_predict=max_tokens,
                )
            )
            elapsed = time.time() - start_time
        except OllamaError as exc:
            st.error(f"⚠️ {exc}")
            st.stop()

        # Real token accounting from Ollama (falls back to wall-clock if absent)
        tokens = stats.get("tokens_generated", 0)
        eval_seconds = stats.get("eval_seconds", 0)
        tps = tokens / eval_seconds if eval_seconds > 0 else (
            tokens / elapsed if elapsed > 0 else 0
        )

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Time Elapsed", f"{elapsed:.2f}s")
        with col2:
            st.metric("Tokens Generated", tokens)
        with col3:
            st.metric("Throughput", f"{tps:.1f} tok/s")
        with col4:
            st.metric("Model", selected_model)

        # Store in history
        st.session_state.inference_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "prompt": prompt,
            "output": output_text,
            "time_elapsed": elapsed,
            "tokens_generated": tokens,
            "throughput": tps
        })

elif page == "Benchmark":
    st.header("⚡ Run Benchmark Suite")
    st.markdown("Evaluate all 3 models on the same standardized prompts.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Models to benchmark:** {', '.join(MODELS.keys())}")
        st.markdown(f"**Prompts:** {len(BENCHMARK_PROMPTS)}")

    with col2:
        if st.button("Start Benchmark", type="primary", use_container_width=True):
            with st.spinner("Running benchmark suite (this may take 5-10 minutes)..."):
                results = run_benchmark()
                st.session_state.benchmark_results = results
                st.success("Benchmark complete!")

    if st.session_state.benchmark_results:
        st.markdown("### Results")
        results_df = json.dumps(st.session_state.benchmark_results, indent=2)

        col1, col2, col3 = st.columns(3)

        for model_name in MODELS.keys():
            if model_name in st.session_state.benchmark_results:
                data = st.session_state.benchmark_results[model_name]
                with col1 if model_name == list(MODELS.keys())[0] else (col2 if model_name == list(MODELS.keys())[1] else col3):
                    st.metric(
                        model_name,
                        f"{data['avg_throughput']:.2f} tok/s",
                        f"Avg: {data['avg_time']:.2f}s"
                    )

elif page == "Comparison":
    st.header("📈 Model Comparison")

    if not st.session_state.benchmark_results:
        st.warning("No benchmark results available. Run the Benchmark first.")
    else:
        # Quality vs Speed comparison
        comparison_data = []
        for model_name, metrics in st.session_state.benchmark_results.items():
            model_cfg = MODELS[model_name]
            comparison_data.append({
                "Model": model_name,
                "Speed (tok/s)": metrics["avg_throughput"],
                "Avg Time (s)": metrics["avg_time"],
                "Quality Rating": model_cfg["quality"],
                "Parameters": model_cfg["params"]
            })

        import pandas as pd
        df = pd.DataFrame(comparison_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Performance Metrics")
            st.dataframe(df, use_container_width=True)

        with col2:
            st.markdown("### Trade-offs Summary")
            for _, row in df.iterrows():
                with st.expander(row["Model"]):
                    st.write(f"**Speed:** {row['Speed (tok/s)']:.2f} tokens/sec")
                    st.write(f"**Quality:** {row['Quality Rating']}/10")
                    st.write(f"**Parameters:** {row['Parameters']}")

# Footer with useful info
st.divider()
st.markdown("""
### 💡 Key Insights
- **Llama 2 7B**: Fast (~30 tok/s), good for real-time apps, privacy-focused
- **Mistral 7B**: Balanced speed & quality, efficient architecture
- **Llama 2 13B**: Best quality, slower (~10-15 tok/s), for complex tasks

**Offline First**: All models run locally on your hardware — no data leaves your machine.
""")
