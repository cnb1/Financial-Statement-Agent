"""
Financial-statement analysis agent — Google ADK implementation.

ADK tools are just plain Python functions: it reads the type hints and
docstring to build the function-calling schema for you, so `run_python`
below IS the tool definition — no separate JSON schema needed like the
OpenAI/Anthropic versions.

ADK runs on an event loop (Runner + SessionService) rather than a manual
"call model -> check for tool_use -> call model again" loop; run_async
handles that orchestration internally.
"""

import asyncio

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from common import run_python as _run_python, SYSTEM_PROMPT

APP_NAME = "financial_agent_app"
USER_ID = "local_user"
SESSION_ID = "session_1"
# Using OpenAI via LiteLLM. LiteLlm reads OPENAI_API_KEY from the environment
# automatically. Swap the model string for any OpenAI chat model you have access to.
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


async def ask(runner: Runner, question: str):
    """Send one message in the ongoing session and print the response.

    Reusing the same runner/session across calls is what makes this a real
    chatbot rather than one-shot Q&A: ADK keeps the conversation history in
    the session, so follow-ups like "what about just 2023?" resolve against
    earlier turns. The pandas DataFrame in common.py's namespace is a plain
    module-level object, so it persists across turns too regardless of
    session state.
    """
    content = types.Content(role="user", parts=[types.Part(text=question)])

    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=content
    ):
        # Surface tool calls/results as they stream in, same as the manual loop
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    print(f"\n[calling tool] {part.function_call.name}({part.function_call.args})\n")
                if getattr(part, "function_response", None):
                    print(f"[tool result]\n{part.function_response.response}\n")

        if event.is_final_response():
            print(f"\nAgent: {event.content.parts[0].text}\n")


async def chat():
    """Interactive REPL: one session persists for the whole conversation."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print("Financial statement chatbot (ADK). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        await ask(runner, question)


if __name__ == "__main__":
    asyncio.run(chat())
