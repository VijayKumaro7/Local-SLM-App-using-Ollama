import streamlit as st
import time
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
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
st.markdown("Compare inference speed and quality across local models running via Ollama.")

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
page = st.sidebar.radio(
    "Navigation",
    ["Generate", "Benchmark", "Comparison", "History", "Live Compare"]
)

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
    st.markdown("Evaluate all configured models on the same standardized prompts.")

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
        results = st.session_state.benchmark_results

        # Per-model metric cards — one column per benchmarked model
        benchmarked = [m for m in MODELS.keys() if m in results]
        if benchmarked:
            metric_cols = st.columns(len(benchmarked))
            for col, model_name in zip(metric_cols, benchmarked):
                data = results[model_name]
                col.metric(
                    model_name,
                    f"{data['avg_throughput']:.2f} tok/s",
                    f"Avg: {data['avg_time']:.2f}s"
                )

            # Interactive throughput chart
            chart_df = pd.DataFrame([
                {
                    "Model": m,
                    "Throughput (tok/s)": results[m]["avg_throughput"],
                    "Avg Time (s)": results[m]["avg_time"],
                }
                for m in benchmarked
            ])
            fig = px.bar(
                chart_df.sort_values("Throughput (tok/s)", ascending=False),
                x="Model",
                y="Throughput (tok/s)",
                color="Model",
                text_auto=".1f",
                title="Average Throughput by Model",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Export the raw results
        st.download_button(
            "⬇️ Download results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="benchmark_results.json",
            mime="application/json",
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

        df = pd.DataFrame(comparison_data)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Performance Metrics")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download comparison (CSV)",
                data=df.to_csv(index=False),
                file_name="model_comparison.csv",
                mime="text/csv",
            )

        with col2:
            st.markdown("### Trade-offs Summary")
            for _, row in df.iterrows():
                with st.expander(row["Model"]):
                    st.write(f"**Speed:** {row['Speed (tok/s)']:.2f} tokens/sec")
                    st.write(f"**Quality:** {row['Quality Rating']}/10")
                    st.write(f"**Parameters:** {row['Parameters']}")

        # Visual comparison
        st.markdown("### Visual Comparison")
        chart1, chart2 = st.columns(2)

        with chart1:
            fig_speed = px.bar(
                df.sort_values("Speed (tok/s)", ascending=False),
                x="Model",
                y="Speed (tok/s)",
                color="Model",
                text_auto=".1f",
                title="Speed (tokens/sec)",
            )
            fig_speed.update_layout(showlegend=False)
            st.plotly_chart(fig_speed, use_container_width=True)

        with chart2:
            fig_tradeoff = px.scatter(
                df,
                x="Speed (tok/s)",
                y="Quality Rating",
                color="Model",
                text="Model",
                title="Speed vs Quality Tradeoff",
            )
            fig_tradeoff.update_traces(textposition="top center")
            fig_tradeoff.update_yaxes(range=[0, 10])
            fig_tradeoff.update_layout(showlegend=False)
            st.plotly_chart(fig_tradeoff, use_container_width=True)

elif page == "History":
    st.header("🕓 Inference History")

    history = st.session_state.inference_history

    if not history:
        st.info("No generations yet — go to **Generate** and run a prompt first.")
    else:
        # Summary row
        avg_tps = sum(e["throughput"] for e in history) / len(history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Generations", len(history))
        c2.metric("Distinct Models", len({e["model"] for e in history}))
        c3.metric("Avg Throughput", f"{avg_tps:.1f} tok/s")

        st.markdown("### Sessions")

        # Build display DataFrame (newest first)
        rows = [
            {
                "Time": e["timestamp"][:19].replace("T", " "),
                "Model": e["model"],
                "Prompt": e["prompt"][:60] + ("…" if len(e["prompt"]) > 60 else ""),
                "Tokens": e["tokens_generated"],
                "Throughput (tok/s)": round(e["throughput"], 1),
                "Duration (s)": round(e["time_elapsed"], 2),
            }
            for e in reversed(history)
        ]
        df_hist = pd.DataFrame(rows)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # CSV export
        st.download_button(
            "⬇️ Download history (CSV)",
            data=df_hist.to_csv(index=False),
            file_name="inference_history.csv",
            mime="text/csv",
        )

        st.markdown("### Details")
        for i, entry in enumerate(reversed(history)):
            label = f"{entry['timestamp'][:19].replace('T', ' ')} — {entry['model']}"
            with st.expander(label):
                st.markdown(f"**Prompt:** {entry['prompt']}")
                st.markdown("**Output:**")
                st.write(entry["output"])
                st.markdown(
                    f"Tokens: **{entry['tokens_generated']}** · "
                    f"Throughput: **{entry['throughput']:.1f} tok/s** · "
                    f"Duration: **{entry['time_elapsed']:.2f}s**"
                )

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.inference_history = []
            st.rerun()

elif page == "Live Compare":
    import threading

    st.header("⚖️ Live Model Comparison")
    st.markdown("Run the same prompt on two models simultaneously and compare outputs.")

    model_keys = list(MODELS.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        model_a = st.selectbox("Model A", model_keys, index=0, key="cmp_a")
    with col_b:
        model_b = st.selectbox("Model B", model_keys,
                               index=min(1, len(model_keys) - 1), key="cmp_b")

    prompt_cmp = st.text_area(
        "Prompt", height=100,
        placeholder="e.g., 'Explain recursion in one paragraph'"
    )
    temp_cmp = st.slider("Temperature", 0.0, 2.0, 0.7, key="cmp_temp")
    tokens_cmp = st.number_input("Max Tokens", 50, 2000, 256, key="cmp_tokens")
    run_cmp = st.button("▶ Run Comparison", type="primary", use_container_width=True)

    if run_cmp and prompt_cmp:
        if model_a == model_b:
            st.warning("Select two different models for a meaningful comparison.")
        else:
            results_cmp: dict = {}

            def _run(name: str) -> None:
                try:
                    results_cmp[name] = inference.generate(
                        model=name,
                        prompt=prompt_cmp,
                        temperature=temp_cmp,
                        num_predict=int(tokens_cmp),
                    )
                except OllamaError as exc:
                    results_cmp[name] = {"error": str(exc)}

            with st.spinner("Running both models in parallel…"):
                t_a = threading.Thread(target=_run, args=(model_a,))
                t_b = threading.Thread(target=_run, args=(model_b,))
                t_a.start(); t_b.start()
                t_a.join();  t_b.join()

            left, right = st.columns(2)

            def _show(col, name: str) -> None:
                r = results_cmp.get(name, {})
                col.subheader(name)
                if "error" in r:
                    col.error(f"⚠️ {r['error']}")
                    return
                tokens = r.get("tokens_generated", 0)
                secs = r.get("eval_seconds", 0)
                tps = tokens / secs if secs > 0 else 0
                col.metric("Throughput", f"{tps:.1f} tok/s")
                col.metric("Tokens", tokens)
                col.metric("Time", f"{secs:.2f}s")
                col.markdown("---")
                col.write(r.get("text", ""))

            _show(left, model_a)
            _show(right, model_b)

# Footer with useful info
st.divider()
st.markdown("""
### 💡 Key Insights
- **Llama 2 7B**: Fast (~30 tok/s), good for real-time apps, privacy-focused
- **Mistral 7B**: Balanced speed & quality, efficient architecture
- **Llama 2 13B**: Best quality, slower (~10-15 tok/s), for complex tasks

**Offline First**: All models run locally on your hardware — no data leaves your machine.
""")
