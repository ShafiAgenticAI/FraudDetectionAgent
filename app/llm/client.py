"""Provider-agnostic LLM client.

Every other module in the app (RAG answering, summarization, obligation
extraction, embeddings) should call the functions in this file instead of
talking to the OpenAI or Gemini SDKs directly. Which provider is actually
used is controlled by the LLM_PROVIDER environment variable ("openai" or
"gemini"), read from app.config.

This keeps provider-specific code in exactly one place.
"""

from __future__ import annotations

from app.config import (
    GEMINI_API_KEY,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
)

_openai_client = None
_gemini_configured = False


# --------------------------------------------------------------------------
# Lazy provider setup
# --------------------------------------------------------------------------


def _get_openai_client():
    global _openai_client

    if _openai_client is None:
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Add it to your .env file "
                "(or set LLM_PROVIDER=gemini and configure GEMINI_API_KEY)."
            )

        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client


def _configure_gemini():
    global _gemini_configured

    import google.generativeai as genai

    if not _gemini_configured:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add it to your .env file "
                "(or set LLM_PROVIDER=openai and configure OPENAI_API_KEY)."
            )

        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_configured = True

    return genai


# --------------------------------------------------------------------------
# Chat completion
# --------------------------------------------------------------------------


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0,
    json_mode: bool = False,
) -> str:
    """Send a system + user prompt to the configured chat provider.

    Returns the raw text of the reply. When json_mode is True, the provider
    is asked to return a JSON object as plain text (the caller is still
    responsible for json.loads-ing it).
    """

    if LLM_PROVIDER == "gemini":
        return _gemini_chat(system_prompt, user_prompt, temperature, json_mode)

    return _openai_chat(system_prompt, user_prompt, temperature, json_mode)


def _openai_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_mode: bool,
) -> str:
    client = _get_openai_client()

    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )

    return response.choices[0].message.content


def _gemini_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_mode: bool,
) -> str:
    genai = _configure_gemini()

    generation_config = {"temperature": temperature}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=GEMINI_CHAT_MODEL,
        system_instruction=system_prompt,
    )

    response = model.generate_content(
        user_prompt,
        generation_config=generation_config,
    )

    return response.text


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (for indexing)."""

    if not texts:
        return []

    if LLM_PROVIDER == "gemini":
        return _gemini_embed(texts, task_type="retrieval_document")

    return _openai_embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string (for search)."""

    if LLM_PROVIDER == "gemini":
        return _gemini_embed([text], task_type="retrieval_query")[0]

    return _openai_embed([text])[0]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()

    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def _gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    genai = _configure_gemini()

    embeddings = []

    for text in texts:
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        embeddings.append(result["embedding"])

    return embeddings
