"""
LLM wrapper — Groq as primary provider.
Retries with alt model if primary fails.
"""
import time
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logger = logging.getLogger(__name__)


def get_llm(model: str = None):
    """Create a Groq LLM instance."""
    from langchain_groq import ChatGroq
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set in .env file."
        )
    return ChatGroq(
        model=model or config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.2,
    )


def _rebind_tools_if_needed(candidate_llm, template_llm):
    """Preserve tool binding when retrying with a fallback model."""
    kwargs = getattr(template_llm, "kwargs", None)
    if not isinstance(kwargs, dict):
        return candidate_llm

    tools = kwargs.get("tools")
    if not tools:
        return candidate_llm

    return candidate_llm.bind_tools(tools)


def safe_llm_call(llm, messages, base_delay: float = 2.0):
    """
    Calls llm.invoke(messages) with retry and model fallback.
    Primary: gemma2-9b-it
    Fallback: llama-3.3-70b-versatile
    """
    models_to_try = [config.GROQ_MODEL, config.GROQ_MODEL_ALT]
    last_error = None

    for model_idx, model_name in enumerate(models_to_try):
        if model_idx == 0:
            current_llm = llm
        else:
            current_llm = _rebind_tools_if_needed(get_llm(model_name), llm)
        label = f"Groq/{model_name}"

        for attempt in range(3):
            try:
                return current_llm.invoke(messages)

            except Exception as e:
                err = str(e).lower()
                is_rate = "429" in err or "rate" in err or \
                          "too many" in err
                is_overload = "503" in err or "overload" in err or \
                              "unavailable" in err
                is_invalid = ("401" in err or "unauthorized" in err) and \
                             "decommission" not in err
                is_not_found = "404" in err or "not found" in err or \
                               "does not exist" in err or \
                               "decommissioned" in err or \
                               "model_decommissioned" in err

                if is_invalid:
                    raise RuntimeError(
                        "GROQ_API_KEY is invalid or unauthorized. "
                        "Check your key in .env"
                    ) from e

                elif is_not_found and model_idx < len(models_to_try) - 1:
                    logger.warning(
                        f"{label} model not found — "
                        f"trying {models_to_try[model_idx + 1]}"
                    )
                    last_error = e
                    break  # try next model

                elif is_rate or is_overload:
                    if attempt < 2:
                        wait = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{label} rate limited "
                            f"(attempt {attempt + 1}/3). "
                            f"Waiting {wait}s..."
                        )
                        time.sleep(wait)
                        last_error = e
                    elif model_idx < len(models_to_try) - 1:
                        logger.warning(
                            f"{label} exhausted — trying alt model."
                        )
                        last_error = e
                        break
                    else:
                        raise RuntimeError(
                            f"All Groq models rate limited. "
                            f"Try again in a minute. Error: {e}"
                        ) from e
                else:
                    logger.error(f"{label} error: {e}")
                    raise

    raise RuntimeError(
        f"All LLM options failed. Last error: {last_error}"
    )
