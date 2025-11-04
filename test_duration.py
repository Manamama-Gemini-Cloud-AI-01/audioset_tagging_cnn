
import argparse
import sys

# --- Installation Check ---
try:
    import torchaudio
    import librosa
    from pydub import AudioSegment
    from mutagen.flac import FLAC
except ImportError as e:
    print(f"Error: A required library is not installed: {e}")
    print("Please install the necessary libraries to run this script:")
    print("pip install torch torchaudio")
    print("pip install librosa")
    print("pip install pydub")
    print("pip install mutagen")
    sys.exit(1)

def get_duration_torchaudio(file_path):
    """Gets duration using torchaudio.info."""
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        return f"Error: {e}"

def get_duration_librosa(file_path):
    """Gets duration using librosa."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        return f"Error: {e}"

def get_duration_pydub(file_path):
    """Gets duration using pydub."""
    try:
        audio = AudioSegment.from_file(file_path)
        return audio.duration_seconds
    except Exception as e:
        return f"Error: {e}"

def get_duration_mutagen(file_path):
    """Gets duration using mutagen."""
    try:
        audio = FLAC(file_path)
        return audio.info.length
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test various methods to get audio file duration.")
    parser.add_argument("file_path", type=str, help="Path to the audio file.")
    args = parser.parse_args()

    print(f"Testing duration for: {args.file_path}\n")

    # Test methods
    duration_torchaudio = get_duration_torchaudio(args.file_path)
    duration_librosa = get_duration_librosa(args.file_path)
    duration_pydub = get_duration_pydub(args.file_path)
    duration_mutagen = get_duration_mutagen(args.file_path)

    # Print results
    print(f"torchaudio: {duration_torchaudio}")
    print(f"librosa:    {duration_librosa}")
    print(f"pydub:      {duration_pydub}")
    print(f"mutagen:    {duration_mutagen}")
