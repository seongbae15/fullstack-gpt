import streamlit as st
import subprocess
from pathlib import Path
from pydub import AudioSegment
import math
from tqdm import tqdm
from glob import glob
import openai

has_transcript = Path("./.cache/files/podcast.txt").exists()


@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcript:
        return
    command = ["ffmpeg", "-i", video_path, "-vn", audio_path, "-y"]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunk_dir):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    min2ms = 60 * 1000
    sep_ms = chunk_size * min2ms
    chunck_length = math.ceil(len(track) / sep_ms)
    name_format = f"0{len(str(chunck_length))}"
    for i in tqdm(range(chunck_length)):
        start_part = i * sep_ms
        end_part = (i + 1) * sep_ms
        file_name = f"podcast_chunk_{i*chunk_size:03d}.mp3"

        chunk = track[start_part:end_part]
        chunk.export(f"{chunk_dir}/{file_name}")


@st.cache_data()
def transcribe_chunks(chunk_dir, destination):
    if has_transcript:
        return
    files = sorted(glob(f"{chunk_dir}/*.mp3"))
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transciprt = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transciprt["text"] + "\n")


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT

Welcom to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])

if video:
    with st.status("Loading Video...") as status:
        video_path = Path(f".cache/files/{video.name}")
        audio_path = video_path.parent.joinpath(f"{video_path.stem}.mp3")
        with open(video_path, "wb") as f:
            f.write(video.read())

        status.update(label="Extracting Audio from video...")
        extract_audio_from_video(video_path, audio_path)

        status.update(label="Cutting audio segments...")

        chunk_size = 10
        chunk_dir = f"./.cache/audio_chunks"
        cut_audio_in_chunks(audio_path, chunk_size, chunk_dir)

        status.update(label="Transcribing audio...")
        transcript_path = video_path.parent.joinpath(f"{video_path.stem}.txt")
        transcribe_chunks(chunk_dir, transcript_path)

    trasnscript_tab, summaray_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with trasnscript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())

    with summaray_tab:
        pass
        # Refine chain.
        # 1. make summary to a doc
        # 2. Update context with this summary
