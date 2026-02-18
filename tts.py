from gtts import gTTS
import io


def synthesize_speech(text: str, lang: str = "en") -> bytes:
    """
    Converts text to speech using gTTS.
    Returns MP3 audio as bytes.
    """
    try:
        if not text or not text.strip():
            return b""

        tts = gTTS(text=text.strip(), lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        mp3_bytes = audio_buffer.read()

        print(f"[TTS] Synthesized {len(mp3_bytes)} bytes for: '{text[:50]}...' " if len(text) > 50 else f"[TTS] Synthesized {len(mp3_bytes)} bytes for: '{text}'")
        return mp3_bytes

    except Exception as e:
        print(f"[TTS] Error: {e}")
        return b""