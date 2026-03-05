# backend/nlp/llm_client.py
# ─────────────────────────────────────────────────────────────────────────────
# LLM Client — Unified Groq + Ollama Wrapper
#
# This module provides a single function, call_llm(), that sends prompts to
# a language model and returns the model's text response.
#
# Architecture:
#   - Primary backend:  Groq Cloud (fast hosted inference, requires API key)
#   - Fallback backend: Ollama    (local inference, requires Ollama running)
#
# Why two backends?
#   Groq is fast and free-tier friendly, but requires internet and an API key.
#   Ollama lets you run models locally — useful when you're offline or when
#   you want to avoid sending sensitive data to a cloud provider.
#   The fallback logic means the pipeline keeps working if one backend fails.
#
# Setup:
#   pip install groq ollama python-dotenv
#   Set GROQ_API_KEY in your .env file (copy .env.example → .env)
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import json
from groq import Groq
from dotenv import load_dotenv

# load_dotenv() reads the .env file in the project root and injects its
# key-value pairs into the current process's environment variables.
# This means os.getenv("GROQ_API_KEY") will work after this call.
load_dotenv()

# ── Model configuration ───────────────────────────────────────────────────────
# These constants define which model to use on each backend.
# You can swap them without touching any other file.
GROQ_MODEL   = "llama-3.3-70b-versatile"   # High-quality 70B model via Groq
OLLAMA_MODEL = "qwen3:8b"                   # Lightweight local model via Ollama

# Initialise the Groq client once at module load time.
# Groq() automatically reads GROQ_API_KEY from the environment.
# We guard with a check so the import doesn't crash if the key is missing.
groq_client = None
if os.getenv("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Tracks which backend is currently active — used by the Streamlit sidebar.
current_backend = "none"


# ── Primary function ──────────────────────────────────────────────────────────

# What this function does:
#   Sends a system prompt and user prompt to the LLM and returns the text reply.
#   It tries Groq first. If Groq fails for any reason, it switches to Ollama.
#
# Parameters:
#   system_prompt  (str)   — instructions that define the model's role/persona
#   user_prompt    (str)   — the actual task or question for this specific call
#   max_tokens     (int)   — upper limit on how many tokens the model may generate
#                            Default: 1000. Lower = faster and cheaper; Higher = longer outputs
#   temperature    (float) — controls randomness. 0.0 = fully deterministic,
#                            1.0 = highly creative. Default: 0.05 (near-deterministic,
#                            good for classification and structured JSON outputs)
#
# Returns:
#   str — the raw text content of the model's reply

def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.05,
) -> str:
    global current_backend

    # ── Attempt 1: Groq ───────────────────────────────────────────────────────
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    # The "system" role sets the model's persona and constraints.
                    # The "user" role contains the specific task for this call.
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60,  # Give up after 60 seconds rather than hanging forever.
            )
            current_backend = "groq"
            # response.choices is a list of possible completions.
            # [0] takes the first (and only) completion.
            # .message.content is the text string we want.
            return response.choices[0].message.content

        except Exception as e:
            # Print the error type so the developer can debug, but don't crash —
            # instead fall through to the Ollama attempt below.
            print(f"⚠ Groq unavailable ({type(e).__name__}: {e}). Trying Ollama...")

    # ── Attempt 2: Ollama (local fallback) ────────────────────────────────────
    try:
        # Import here so the app doesn't crash at startup if ollama isn't installed.
        import ollama

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens.
            },
        )
        current_backend = "ollama"
        # Ollama returns a dict; the message content lives at ["message"]["content"].
        return response["message"]["content"]

    except Exception as e:
        # Both backends failed. Re-raise with a clear diagnosis message.
        raise RuntimeError(
            f"Both Groq and Ollama failed.\n"
            f"  • Groq requires internet + a valid GROQ_API_KEY in your .env file.\n"
            f"  • Ollama requires the Ollama app to be running locally.\n"
            f"  Ollama error: {e}"
        )


# ── Retry wrapper ─────────────────────────────────────────────────────────────

# What this function does:
#   Wraps call_llm() with automatic retry logic using exponential back-off.
#   If the first attempt fails, it waits 1 second and tries again.
#   If the second attempt fails, it waits 2 seconds. Then 4, and so on.
#   This handles transient network errors and temporary API rate limits.
#
# Parameters:
#   system_prompt  (str)   — same as call_llm()
#   user_prompt    (str)   — same as call_llm()
#   max_tokens     (int)   — same as call_llm()
#   temperature    (float) — same as call_llm()
#   retries        (int)   — how many total attempts before giving up. Default: 3
#
# Returns:
#   str — the raw text content of the model's reply

def call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.05,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            return call_llm(system_prompt, user_prompt, max_tokens, temperature)

        except RuntimeError:
            # RuntimeError means both backends are down — no point retrying.
            raise

        except Exception as e:
            if attempt < retries - 1:
                # Exponential back-off: wait 2^attempt seconds (1, 2, 4…)
                # before the next retry.
                wait_seconds = 2 ** attempt
                print(f"  Attempt {attempt + 1} failed ({e}). Retrying in {wait_seconds}s…")
                time.sleep(wait_seconds)
            else:
                # Final attempt also failed — give up and propagate the error.
                raise


# ── JSON extraction helper ─────────────────────────────────────────────────────

# What this function does:
#   Attempts to parse the model's reply as JSON.
#   LLMs sometimes wrap JSON in markdown code fences (```json … ```) or add
#   preamble text. This function strips those and returns a Python dict/list.
#
# Parameters:
#   raw_text   (str)  — the raw string returned by call_llm()
#   fallback   (any)  — value to return if parsing fails (default: empty dict)
#
# Returns:
#   dict | list | any — the parsed Python object, or fallback on failure

def parse_llm_json(raw_text: str, fallback=None) -> dict:
    if fallback is None:
        fallback = {}

    # Strip markdown code fences if the model wrapped the JSON in them.
    # e.g. "```json\n{...}\n```" → "{...}"
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Find the first newline after the opening fence and the last closing fence.
        first_newline = cleaned.find("\n")
        last_fence    = cleaned.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            cleaned = cleaned[first_newline:last_fence].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Return the fallback so the pipeline doesn't crash on a bad response.
        print(f"⚠ Could not parse LLM response as JSON. Raw text:\n{raw_text[:300]}")
        return fallback


# ── Status helper ──────────────────────────────────────────────────────────────

def get_active_backend() -> str:
    """Returns the currently active backend name — used by the Streamlit sidebar."""
    return current_backend
