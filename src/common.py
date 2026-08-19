"""
Shared setup used by both the OpenAI agent and the Google ADK agent:
- resolves paths to resources/ (sibling of src/)
- loads .env from resources/
- loads the CSV into a DataFrame
- exposes run_python(code) which executes against that DataFrame

Keeping this in one place means both agents analyze the exact same data
the exact same way — only the "agent framework glue" differs per provider.
"""

import io
import os
import contextlib
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths: src/ and resources/ are siblings under the project root.
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")

load_dotenv(os.path.join(RESOURCES_DIR, ".env"))

CSV_PATH = os.environ.get(
    "CSV_PATH", os.path.join(RESOURCES_DIR, "financial_statement.csv")
)
# CSV_PATH in .env is relative to src/ by convention; resolve if not absolute
if not os.path.isabs(CSV_PATH):
    CSV_PATH = os.path.normpath(os.path.join(SRC_DIR, CSV_PATH))

# ---------------------------------------------------------------------------
# Data + persistent execution namespace
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

namespace = {"pd": pd, "df": df}

SCHEMA_PREVIEW = (
    f"Columns: {list(df.columns)}\n"
    f"Dtypes:\n{df.dtypes}\n"
    f"Full data (small enough to show in full):\n{df.to_string(index=False)}"
)

SYSTEM_PROMPT = (
    "You are a financial analyst. You can inspect and analyze an income "
    "statement using the run_python tool, which executes code against a "
    "pandas DataFrame called `df` (one row per fiscal year).\n\n"
    f"{SCHEMA_PREVIEW}\n\n"
    "When asked about margins, growth rates, or trends, compute them "
    "explicitly in code (e.g. gross margin = gross_profit / revenue, "
    "YoY growth = pct_change()) rather than estimating from memory. "
    "Give a clear, concise final answer with the key numbers."
    "Your personality is also a very angry and overworked analyst "
    "and your response tone should reflect an angry overworked feeling."
)

def run_python(code: str) -> str:
    """Execute code in the shared namespace, capture stdout, return it as text.

    NOTE: this uses exec() with no sandboxing — fine for local prototyping,
    not safe for untrusted input in production. Swap for a subprocess with a
    timeout or a container if this needs to run against arbitrary/remote
    input.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)
    except Exception as e:
        return f"ERROR: {e}"
    output = buf.getvalue().strip()
    return output if output else "(code ran, but printed nothing — use print())"
