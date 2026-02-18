import whisper
import numpy as np
import io
import tempfile
import os

# Load Whisper base model once at startup (CPU mode)
print("[STT] Loading Whisper base model on CPU...")
model = whisper.load_model("base", device="cpu")
print("[STT] Whisper model loaded successfully.")


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Accepts raw audio bytes (webm/wav from browser),
    saves to temp file, transcribes with Whisper base.
    Returns transcribed text string.
    """
    try:
        # Write to a temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe
        result = model.transcribe(tmp_path, fp16=False, language="en")
        text = result["text"].strip()

        # Cleanup
        os.remove(tmp_path)

        print(f"[STT] Transcribed: {text}")
        return text

    except Exception as e:
        print(f"[STT] Error during transcription: {e}")
        return ""