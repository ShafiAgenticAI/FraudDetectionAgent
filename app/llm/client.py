"""Provider-agnostic LLM client.

Every other module in the app (RAG answering, summarization, obligation
extraction, embeddings) should call the functions in this file instead of
talking to the OpenAI or Gemini SDKs directly. Which provider is actually
used is controlled by the LLM_PROVIDER environment variable ("openai" or
"gemini"), read from app.config.

This keeps provider-specific code in exactly one place.
"""

from __future__ import annotations

import json

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    BEDROCK_CHAT_MODEL,
    BEDROCK_EMBEDDING_MODEL,
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
_bedrock_client = None


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


def _get_bedrock_client():
    global _bedrock_client

    if _bedrock_client is None:
        import boto3

        kwargs = {"region_name": AWS_REGION}

        # If explicit creds are set, pass them; otherwise let boto3 fall
        # back to its normal default chain (env vars already named
        # AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN are
        # picked up automatically, as are ~/.aws/credentials or an IAM
        # role, so explicit passing here is just belt-and-suspenders for
        # the .env-driven config this app otherwise uses).
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
            if AWS_SESSION_TOKEN:
                kwargs["aws_session_token"] = AWS_SESSION_TOKEN

        _bedrock_client = boto3.client("bedrock-runtime", **kwargs)

    return _bedrock_client


def _strip_json_fences(text: str) -> str:
    """Some Bedrock models wrap JSON in ```json ... ``` even when asked
    not to; strip that so json.loads() in the caller doesn't choke."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


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

    if LLM_PROVIDER == "bedrock":
        return _bedrock_chat(system_prompt, user_prompt, temperature, json_mode)

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


def _bedrock_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_mode: bool,
) -> str:
    client = _get_bedrock_client()

    prompt = user_prompt

    if json_mode:
        prompt += (
            "\n\nRespond with ONLY a valid JSON object and nothing else "
            "-- no markdown code fences, no commentary before or after it."
        )

    try:
        response = client.converse(
            modelId=BEDROCK_CHAT_MODEL,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": temperature},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock chat call failed for model '{BEDROCK_CHAT_MODEL}' in "
            f"region '{AWS_REGION}': {exc}. If this is an AccessDenied or "
            "ValidationException, enable model access for this model under "
            "Bedrock console -> Model access, and confirm your "
            "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION are correct."
        ) from exc

    text = response["output"]["message"]["content"][0]["text"]

    if json_mode:
        text = _strip_json_fences(text)

    return text


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents (for indexing)."""

    if not texts:
        return []

    if LLM_PROVIDER == "gemini":
        return _gemini_embed(texts, task_type="retrieval_document")

    if LLM_PROVIDER == "bedrock":
        return _bedrock_embed(texts)

    return _openai_embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single query string (for search)."""

    if LLM_PROVIDER == "gemini":
        return _gemini_embed([text], task_type="retrieval_query")[0]

    if LLM_PROVIDER == "bedrock":
        return _bedrock_embed([text])[0]

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


def _bedrock_embed(texts: list[str]) -> list[list[float]]:
    client = _get_bedrock_client()
    embeddings = []

    for text in texts:
        body = json.dumps({"inputText": text})

        try:
            response = client.invoke_model(
                modelId=BEDROCK_EMBEDDING_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock embedding call failed for model "
                f"'{BEDROCK_EMBEDDING_MODEL}' in region '{AWS_REGION}': "
                f"{exc}. Confirm model access is enabled and your AWS "
                "credentials/region are correct."
            ) from exc

        payload = json.loads(response["body"].read())
        embeddings.append(payload["embedding"])

    return embeddings
