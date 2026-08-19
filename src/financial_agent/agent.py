"""
Financial-statement analysis agent — definition only.

`adk web` discovers agents by looking for a `root_agent` inside an
agent.py file within a package folder (this one). The interactive chat
loop from adk_agent.py isn't needed here — the ADK web UI provides its
own chat interface and calls this agent directly.
"""

import sys
import os

# common.py lives one level up in src/ — add it to the path so imports work
# regardless of where `adk web` is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from common import run_python as _run_python, SYSTEM_PROMPT

# Using OpenAI via LiteLLM instead of a native Gemini model. LiteLlm reads
# OPENAI_API_KEY from the environment automatically — same var OpenAI's own
# SDK uses. Swap the model string for any OpenAI chat model you have access to.
MODEL = LiteLlm(model="openai/gpt-4.1")


def run_python(code: str) -> str:
    """Execute Python code against the pandas DataFrame `df`, which holds
    annual financial statement data (revenue, expenses, income line items)
    indexed by `year`. Use print() to output whatever you want to see back.

    Args:
        code: Python code to execute.

    Returns:
        Whatever the code printed to stdout, or an error message.
    """
    return _run_python(code)


root_agent = Agent(
    name="financial_analyst",
    model=MODEL,
    description="Answers questions about an annual income statement using pandas code execution.",
    instruction=SYSTEM_PROMPT,
    tools=[run_python],
)
