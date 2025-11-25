import subprocess
import whisper

def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", "_audio.wav")
    cmd = f"ffmpeg -y -i '{video_path}' -ar 16000 -ac 1 '{audio_path}'"
    subprocess.call(cmd, shell=True)
    return audio_path


def transcribe(audio_path):
    model = whisper.load_model("small")
    return model.transcribe(audio_path)["text"]
