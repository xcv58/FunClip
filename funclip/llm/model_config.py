"""Shared model defaults and narrowly scoped provider compatibility."""


DEFAULT_LLM_MODEL = "gpt-6-luna"
LLM_MODEL_CHOICES = (
    DEFAULT_LLM_MODEL,
    "gpt-4o-mini",
    "gpt-4o",
    "gemini-1.5-flash",
    "Custom",
)


def litellm_model_name(model: str) -> str:
    """Route Luna explicitly for LiteLLM versions predating its release."""
    return "openai/gpt-6-luna" if model == "gpt-6-luna" else model


def completion_options(model: str, *, max_tokens: int | None = None) -> dict:
    options = {"model": litellm_model_name(model)}
    is_luna = model in {"gpt-6-luna", "openai/gpt-6-luna"}
    if is_luna:
        # These focused text tasks do not need a separate reasoning budget.
        # Older LiteLLM versions require explicit permission for this parameter.
        options.update(
            reasoning_effort="none",
            allowed_openai_params=["reasoning_effort"],
        )
    if max_tokens is not None:
        options["max_completion_tokens" if is_luna else "max_tokens"] = max_tokens
    return options
