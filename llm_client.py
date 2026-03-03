#import packages
import os
import json
from groq import Groq
import ollama
from dotenv import load_dotenv

load_dotenv()


#CONFIGURATION

GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "qwen3:8b"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Track which backend is being used
current_backend = "groq"

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
    """
    This function is a unified call of the LLM with an automatic fallback:
    1. Try Groq (8b model)
    2. If Groq fails -> switch to local Ollama
    3. Retruns text output.
    """
    global current_backend

    if os.getenv("GROQ_API_KEY"):
        try:
            response = groq_client.chat.completions.create(
                model = GROQ_MODEL,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens = max_tokens, #sets an upper limit on the number of tokens the model generates in response
                temperature = temperature, #controls randomness and probability distribution of the next token
                timeout = 30 #30-second timeout before giving up.
            )
            current_backeend = "groq"
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ Groq unavailable ({type(e).__name__}). Switching to local Ollama...")
    
    #------------Fallback: Local Ollama-------------------
    try:
        response = ollama.chat(
            model = OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        current_backend="ollama"
        return response["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            f"Both Groq and Ollama failed. Groq needs internet; "
            "Ollama needs to be running. Check both. \n"
            f"Ollama error: {e}"
        )

def get_active_backend() -> str:
    #Returns the current active backend - useful for Streamlit status display.
    return current_backend

def call_llm_with_retry(system_prompt, user_prompt, max_tokens = 1000, temperature = 0.1, retries = 3):
    #Full retry wrapper with backoff - to be used everywhere in the pipeline.
    for attempt in range(retries):
        try:
            return call_llm(system_prompt, user_prompt, max_tokens)
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
                    