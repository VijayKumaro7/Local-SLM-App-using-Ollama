import streamlit as st
import time
import json
from datetime import datetime
from inference import OllamaInference
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

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Generate", "Benchmark", "Comparison"])

# Model selector
st.sidebar.markdown("### Model Settings")
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(MODELS.keys()),
    help="Choose which model to use for inference"
)

model_info = MODELS[selected_model]
st.sidebar.info(
    f"**{selected_model}**\n\n"
    f"Params: {model_info['params']}\n"
    f"Est. Speed: {model_info['speed']}\n"
    f"Est. Quality: {model_info['quality']}"
)

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
        inference = OllamaInference()

        with st.spinner(f"Generating with {selected_model}..."):
            start_time = time.time()
            response = inference.generate(
                model=selected_model,
                prompt=prompt,
                temperature=temperature,
                num_predict=max_tokens
            )
            elapsed = time.time() - start_time

        # Display results
        st.markdown("### Generated Output")
        st.write(response["text"])

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Time Elapsed", f"{elapsed:.2f}s")
        with col2:
            tokens = len(response["text"].split())
            st.metric("Tokens Generated", tokens)
        with col3:
            tps = tokens / elapsed if elapsed > 0 else 0
            st.metric("Throughput", f"{tps:.1f} tok/s")
        with col4:
            st.metric("Model", selected_model)

        # Store in history
        st.session_state.inference_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": selected_model,
            "prompt": prompt,
            "output": response["text"],
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
