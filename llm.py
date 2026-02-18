import os
import httpx
import json
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "ARIA Voice Bot"
}

SYSTEM_PROMPT = """You are ARIA — an intelligent, concise, and friendly AI voice assistant.
You are having a spoken conversation, so:
- Keep responses natural and conversational
- Avoid markdown, bullet points, or special formatting
- Speak in clear, complete sentences
- Be helpful, warm, and direct
- Keep answers reasonably brief unless asked for detail
"""


async def stream_llm_response(
    messages: list,
) -> AsyncGenerator[str, None]:
    """
    Streams LLM response from OpenRouter token by token.
    Yields complete sentences as they form.
    """
    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages
    ]

    payload = {
        "model": MODEL,
        "messages": full_messages,
        "max_tokens": 400,
        "stream": True,
        "temperature": 0.7,
    }

    sentence_buffer = ""
    sentence_enders = {".", "!", "?", "\n"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            OPENROUTER_URL,
            headers=HEADERS,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Strip "data: "

                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"]
                    token = delta.get("content", "")

                    if not token:
                        continue

                    sentence_buffer += token

                    # Check if we have a complete sentence
                    for ender in sentence_enders:
                        if ender in sentence_buffer:
                            parts = sentence_buffer.split(ender)
                            # Yield all complete sentences
                            for part in parts[:-1]:
                                sentence = part.strip() + ender
                                if sentence.strip() and len(sentence.strip()) > 2:
                                    print(f"[LLM] Sentence: {sentence}")
                                    yield sentence
                            # Keep the remainder
                            sentence_buffer = parts[-1]
                            break

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # Yield any remaining text
    if sentence_buffer.strip():
        print(f"[LLM] Final chunk: {sentence_buffer.strip()}")
        yield sentence_buffer.strip() + "."