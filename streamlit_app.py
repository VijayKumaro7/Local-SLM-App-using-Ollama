"""Deployment entry point for Streamlit Community Cloud.

Streamlit Cloud auto-detects a file named ``streamlit_app.py`` as the main
module. This thin wrapper simply runs the real application in ``app.py`` so the
app logic stays in one place.

To deploy:
1. Push this repo to GitHub.
2. On https://share.streamlit.io create a new app pointing at this file
   (``streamlit_app.py``) on your branch.
3. (Optional) Set an ``OLLAMA_HOST`` secret to a reachable Ollama server.
   Streamlit Cloud has no Ollama/GPU, so the default ``localhost:11434`` will
   not work there — point it at a remote/tunneled Ollama instance for live
   inference. The UI loads either way.
"""
import os
import runpy

_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
runpy.run_path(_APP, run_name="__main__")
