# contextual_insight/colab_app.py

from IPython.display import display, HTML, clear_output, Markdown
import pandas as pd, numpy as np, time, difflib, base64, requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from google.colab import output
except ImportError:
    class _Dummy:
        def register_callback(self,*a,**k): pass
    output = _Dummy()

import gdown

# — Helpers & state —
STOP = set(ENGLISH_STOP_WORDS) | {
    "customer","client","account","issue","feature","time","good","bad"
}

PROMPTS = {
    "Marketing": "...",   # use your templates here
    "Sales":     "...",
    "Support":   "..."
}

state = {"df": None}

# clustering, callbacks, UI functions (same code as before)…
def compute_state(texts): ...
def top_keywords(state,top_n=8): ...
def handle_files(data_dict): ...
def handle_url(url): ...
def do_analysis(col,method,k,atype): ...
def step1(): ...
def step2(): ...

def launch():
    """Kicks off the Colab drag-drop UI."""
    step1()

output.register_callback("notebook.handleFiles",  handle_files)
output.register_callback("notebook.handleURL",    handle_url)
output.register_callback("notebook.doAnalysis",   do_analysis)
