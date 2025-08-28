import whisper
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import nltk
import os

# Download NLTK tokenizer if not already present
nltk.download('punkt')

# Load Whisper model
model = whisper.load_model("base")

# Load audio file (MP3 or WAV)
def convert_audio_to_wav(input_path, output_path="converted.wav"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error converting audio: {e}")

# Transcribe audio
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result["segments"]  # includes timestamps
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")

# Segment topics using NLP
def segment_topics(transcript):
    summarizer = pipeline("summarization")
    sentences = sent_tokenize(transcript)
    chunks = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"(Summarization failed: {e})")
    return summaries

# Generate timestamped topics
def generate_timestamped_topics(segments):
    full_text = " ".join([seg["text"] for seg in segments])
    summaries = segment_topics(full_text)
    timestamps = [
        segments[i * 5]["start"] if i * 5 < len(segments) else segments[-1]["start"]
        for i in range(len(summaries))
    ]
    return list(zip(timestamps, summaries))

# Main function
def process_podcast(file_path):
    try:
        wav_path = convert_audio_to_wav(file_path)
        segments = transcribe_audio(wav_path)
        timestamped_topics = generate_timestamped_topics(segments)
        for ts, topic in timestamped_topics:
            mins = int(ts // 60)
            secs = int(ts % 60)
            print(f"[{mins:02d}:{secs:02d}] {topic}")
    except Exception as e:
        print(f"Processing failed: {e}")

# Example usage
process_podcast("your_podcast.mp3")