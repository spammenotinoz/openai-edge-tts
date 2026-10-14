# tts_handler.py

import edge_tts
import asyncio
import tempfile
import subprocess
import os
import time
from pathlib import Path

from utils import DETAILED_ERROR_LOGGING
from config import DEFAULT_CONFIGS

# Microsoft's TTS service occasionally returns no audio on a perfectly valid
# request (rany2/edge-tts#473). Rather than surfacing that as a hard 500, retry
# the whole request a few times with a short backoff before giving up.
TTS_MAX_RETRIES = int(os.getenv("TTS_MAX_RETRIES", "3"))
TTS_RETRY_BACKOFF_SECONDS = float(os.getenv("TTS_RETRY_BACKOFF_SECONDS", "1.5"))

# Language default (environment variable)
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])

# OpenAI voice names mapped to edge-tts equivalents
voice_mapping = {
    'alloy': 'en-US-JennyNeural',
    'ash': 'en-US-AndrewNeural',
    'ballad': 'en-GB-ThomasNeural',
    'coral': 'en-AU-NatashaNeural',
    'echo': 'en-US-GuyNeural',
    'fable': 'en-GB-SoniaNeural',
    'nova': 'en-US-AriaNeural',
    'onyx': 'en-US-EricNeural',
    'sage': 'en-US-JennyNeural',
    'shimmer': 'en-US-EmmaNeural',
    'verse': 'en-US-BrianNeural',
}

model_data = [
        {"id": "tts-1", "name": "Text-to-speech v1"},
        {"id": "tts-1-hd", "name": "Text-to-speech v1 HD"},
        {"id": "gpt-4o-mini-tts", "name": "GPT-4o mini TTS"}
    ]

def is_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def _is_transient_tts_error(exc):
    """Only retry errors that are likely to succeed on a fresh attempt.

    ``NoAudioReceived`` is the known intermittent failure (rany2/edge-tts#473);
    connection-level errors may also clear up on a retry. FFmpeg conversion
    failures and malformed-request errors are not retried.
    """
    if isinstance(exc, edge_tts.exceptions.NoAudioReceived):
        return True
    try:
        import aiohttp

        if isinstance(exc, (aiohttp.ClientError, aiohttp.ClientConnectionError)):
            return True
    except Exception:
        pass
    return False


async def _generate_audio_stream(text, voice, speed):
    """Generate streaming TTS audio using edge-tts."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is

    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    # The upstream service occasionally returns no audio on a valid request, so
    # retry the whole stream a few times before surfacing the error.
    last_exc = None
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            # Create the communicator for streaming
            communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)

            # Stream the audio data
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            return
        except Exception as e:
            last_exc = e
            if attempt >= TTS_MAX_RETRIES or not _is_transient_tts_error(e):
                raise
            print(
                f"TTS streaming attempt {attempt}/{TTS_MAX_RETRIES} failed "
                f"({type(e).__name__}); retrying in "
                f"{TTS_RETRY_BACKOFF_SECONDS * attempt}s"
            )
            time.sleep(TTS_RETRY_BACKOFF_SECONDS * attempt)

def generate_speech_stream(text, voice, speed=1.0):
    """Yield audio chunks from the async edge-tts stream (synchronous generator).

    ``asyncio.run`` cannot consume an async generator directly, so a background
    thread runs the event loop and bridges chunks (and any final exception)
    back through a thread-safe queue.
    """
    import threading
    import queue as queue_module

    chunk_queue = queue_module.Queue()

    async def _produce():
        try:
            async for chunk in _generate_audio_stream(text, voice, speed):
                chunk_queue.put(chunk)
        except Exception as e:
            chunk_queue.put(e)
        finally:
            chunk_queue.put(None)  # Sentinel marking end of stream

    threading.Thread(target=lambda: asyncio.run(_produce()), daemon=True).start()

    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            raise chunk
        yield chunk

async def _generate_audio(text, voice, response_format, speed):
    """Generate TTS audio and optionally convert to a different format."""
    # Determine if the voice is an OpenAI-compatible voice or a direct edge-tts voice
    edge_tts_voice = voice_mapping.get(voice, voice)  # Use mapping if in OpenAI names, otherwise use as-is

    # Generate the TTS output in mp3 format first
    temp_mp3_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_mp3_path = temp_mp3_file_obj.name

    # Convert speed to SSML rate format
    try:
        speed_rate = speed_to_rate(speed)  # Convert speed value to "+X%" or "-X%"
    except Exception as e:
        print(f"Error converting speed: {e}. Defaulting to +0%.")
        speed_rate = "+0%"

    # Generate the MP3 file. The upstream service occasionally returns no audio
    # on a valid request, so retry a few times before surfacing the error.
    last_exc = None
    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            communicator = edge_tts.Communicate(text=text, voice=edge_tts_voice, rate=speed_rate)
            await communicator.save(temp_mp3_path)
            break
        except Exception as e:
            last_exc = e
            if attempt >= TTS_MAX_RETRIES or not _is_transient_tts_error(e):
                raise
            print(
                f"TTS attempt {attempt}/{TTS_MAX_RETRIES} failed "
                f"({type(e).__name__}); retrying in "
                f"{TTS_RETRY_BACKOFF_SECONDS * attempt}s"
            )
            time.sleep(TTS_RETRY_BACKOFF_SECONDS * attempt)
    temp_mp3_file_obj.close() # Explicitly close our file object for the initial mp3

    # If the requested format is mp3, return the generated file directly
    if response_format == "mp3":
        return temp_mp3_path

    # Check if FFmpeg is installed
    if not is_ffmpeg_installed():
        print("FFmpeg is not available. Returning unmodified mp3 file.")
        return temp_mp3_path # Return the original mp3 path, it won't be cleaned by this function

    # Create a new temporary file for the converted output
    converted_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}")
    converted_path = converted_file_obj.name
    converted_file_obj.close() # Close file object, ffmpeg will write to the path

    # Build the FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", temp_mp3_path,  # Input file path
        "-c:a", {
            "aac": "aac",
            "mp3": "libmp3lame",
            "wav": "pcm_s16le",
            "opus": "libopus",
            "flac": "flac"
        }.get(response_format, "aac"),  # Default to AAC if unknown
    ]

    if response_format != "wav":
        ffmpeg_command.extend(["-b:a", "192k"])

    ffmpeg_command.extend([
        "-f", {
            "aac": "mp4",  # AAC in MP4 container
            "mp3": "mp3",
            "wav": "wav",
            "opus": "ogg",
            "flac": "flac"
        }.get(response_format, response_format),  # Default to matching format
        "-y",  # Overwrite without prompt
        converted_path  # Output file path
    ])

    try:
        # Run FFmpeg command and ensure no errors occur
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Clean up potentially created (but incomplete) converted file
        Path(converted_path).unlink(missing_ok=True)
        # Clean up the original mp3 file as well, since conversion failed
        Path(temp_mp3_path).unlink(missing_ok=True)
        
        if DETAILED_ERROR_LOGGING:
            error_message = f"FFmpeg error during audio conversion. Command: '{' '.join(e.cmd)}'. Stderr: {e.stderr.decode('utf-8', 'ignore')}"
            print(error_message) # Log for server-side diagnosis
        else:
            error_message = f"FFmpeg error during audio conversion: {e}"
            print(error_message) # Log a simpler message
        raise RuntimeError(f"FFmpeg error during audio conversion: {e}") # The raised error will still have details via e

    # Clean up the original temporary file (original mp3) as it's now converted
    Path(temp_mp3_path).unlink(missing_ok=True)

    return converted_path

def generate_speech(text, voice, response_format, speed=1.0):
    return asyncio.run(_generate_audio(text, voice, response_format, speed))

def get_models():
    return model_data

def get_models_formatted():
    return [{ "id": x["id"] } for x in model_data]

def get_voices_formatted():
    return [{ "id": k, "name": v } for k, v in voice_mapping.items()]

async def _get_voices(language=None):
    # List all voices, filter by language if specified
    all_voices = await edge_tts.list_voices()
    language = language or DEFAULT_LANGUAGE  # Use default if no language specified
    filtered_voices = [
        {"name": v['ShortName'], "gender": v['Gender'], "language": v['Locale']}
        for v in all_voices if language == 'all' or language is None or v['Locale'] == language
    ]
    return filtered_voices

def get_voices(language=None):
    return asyncio.run(_get_voices(language))

def speed_to_rate(speed: float) -> str:
    """
    Converts a multiplicative speed value to the edge-tts "rate" format.
    
    Args:
        speed (float): The multiplicative speed value (e.g., 1.5 for +50%, 0.5 for -50%).
    
    Returns:
        str: The formatted "rate" string (e.g., "+50%" or "-50%").
    """
    if speed < 0 or speed > 2:
        raise ValueError("Speed must be between 0 and 2 (inclusive).")

    # Convert speed to percentage change
    percentage_change = (speed - 1) * 100

    # Format with a leading "+" or "-" as required
    return f"{percentage_change:+.0f}%"
