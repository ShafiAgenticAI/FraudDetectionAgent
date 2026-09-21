"""Embedding helpers.

This module keeps the same public interface (embed_texts / embed_query) that
the rest of the app already imports from app.rag.embeddings, but the actual
work is delegated to app.llm.client, which picks OpenAI or Gemini based on
the LLM_PROVIDER setting in app.config.
"""

from app.llm.client import embed_query, embed_texts

__all__ = ["embed_texts", "embed_query"]
