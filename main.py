import asyncio
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor

from stt import transcribe_audio
from llm import stream_llm_response
from tts import synthesize_speech

app = FastAPI(title="ARIA Voice Bot")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Thread pool for blocking operations (Whisper + gTTS)
executor = ThreadPoolExecutor(max_workers=2)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    # Conversation history for multi-turn memory
    conversation_history = []

    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))

    async def send_audio(mp3_bytes: bytes, sentence: str):
        """Send audio chunk as base64 with the sentence text."""
        if mp3_bytes:
            audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
            await send_json({
                "type": "audio_chunk",
                "audio": audio_b64,
                "text": sentence
            })

    try:
        while True:
            # Receive audio bytes from client
            message = await websocket.receive()

            if "bytes" in message:
                audio_bytes = message["bytes"]
                print(f"[WS] Received {len(audio_bytes)} bytes of audio")

                # ── STEP 1: Status → Transcribing
                await send_json({"type": "status", "state": "transcribing"})

                # ── STEP 2: Whisper STT (blocking → thread pool)
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    executor,
                    transcribe_audio,
                    audio_bytes
                )

                if not transcript:
                    await send_json({"type": "status", "state": "idle"})
                    await send_json({"type": "error", "message": "Could not transcribe audio. Please try again."})
                    continue

                # Send transcript to client
                await send_json({
                    "type": "transcript",
                    "text": transcript,
                    "role": "user"
                })

                # Add to history
                conversation_history.append({
                    "role": "user",
                    "content": transcript
                })

                # ── STEP 3: Status → Thinking
                await send_json({"type": "status", "state": "thinking"})

                # ── STEP 4: Stream LLM + TTS sentence by sentence
                await send_json({"type": "status", "state": "speaking"})

                full_response = ""

                async for sentence in stream_llm_response(conversation_history):
                    full_response += sentence + " "

                    # Send text token to UI immediately
                    await send_json({
                        "type": "transcript",
                        "text": sentence,
                        "role": "assistant",
                        "streaming": True
                    })

                    # TTS in thread pool (gTTS is blocking)
                    mp3_bytes = await loop.run_in_executor(
                        executor,
                        synthesize_speech,
                        sentence
                    )

                    # Send audio chunk
                    await send_audio(mp3_bytes, sentence)

                # Add full response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": full_response.strip()
                })

                # Signal response complete
                await send_json({"type": "response_complete"})
                await send_json({"type": "status", "state": "idle"})

            elif "text" in message:
                # Handle control messages
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "clear_history":
                        conversation_history.clear()
                        await send_json({"type": "history_cleared"})
                        print("[WS] Conversation history cleared")
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await send_json({"type": "error", "message": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)