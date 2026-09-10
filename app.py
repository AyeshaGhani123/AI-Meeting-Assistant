# ============================================================
# AI MEETING ASSISTANT
#
# Audio upload / microphone recording
# ↓
# FFmpeg 16 kHz mono
# ↓
# NeMo Sortformer
# ↓
# Faster-Whisper FULL AUDIO TRANSCRIPTION
# ↓
# Match Whisper timestamps to speakers
# ↓
# Merge consecutive speaker speech
# ↓
# Emotion Classification PER SPEAKER
# ↓
# Summary + Notes
# ↓
# Email Draft
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import re
import uuid
import json
import time
import wave
import shutil
import threading
import subprocess

from pathlib import Path

import numpy as np
import torch
import gradio as gr

from faster_whisper import WhisperModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from nemo.collections.asr.models import (
    SortformerEncLabelModel
)

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "AI Meeting Assistant"

BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "meeting_outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_DIR = BASE_DIR / "meeting_temp"
TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CUSTOM_TEMP = BASE_DIR / "nemo_temp"
CUSTOM_TEMP.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL NAMES
# ============================================================

NEMO_MODEL_NAME = (
    "nvidia/diar_sortformer_4spk-v1"
)

WHISPER_MODEL_NAME = "small"

SUMMARY_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

# NEW
EMOTION_MODEL_NAME = (
    "j-hartmann/emotion-english-distilroberta-base"
)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# ============================================================
# 4. PRINT ENVIRONMENT
# ============================================================

print()

print("=" * 70)
print("AI MEETING ASSISTANT")
print("=" * 70)

print(
    "Python:",
    sys.version.split()[0]
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "Selected device:",
    DEVICE
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    print(
        "Running on CPU."
    )

print("=" * 70)


# ============================================================
# 5. FFMPEG CHECK
# ============================================================

print()

print("=" * 70)
print("CHECKING FFMPEG")
print("=" * 70)

try:

    result = subprocess.run(
        [
            "ffmpeg",
            "-version"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg is installed but not working."
        )

    print(
        result.stdout.splitlines()[0]
    )

except FileNotFoundError:

    raise RuntimeError(
        "\nFFmpeg was not found.\n\n"
        "Install FFmpeg and add it to Windows PATH."
    )


# ============================================================
# 6. LOAD NEMO SORTFORMER
# ============================================================

print()

print("=" * 70)
print("LOADING NEMO SORTFORMER")
print("=" * 70)

print(
    "Model:",
    NEMO_MODEL_NAME
)

nemo_model = (
    SortformerEncLabelModel.from_pretrained(
        model_name=NEMO_MODEL_NAME
    )
)

nemo_model = nemo_model.to(
    DEVICE
)

nemo_model.eval()

print(
    "NeMo Sortformer ready."
)


# ============================================================
# 7. LOAD FASTER WHISPER
# ============================================================

print()

print("=" * 70)
print("LOADING FASTER-WHISPER")
print("=" * 70)

print(
    "Model:",
    WHISPER_MODEL_NAME
)

if DEVICE == "cuda":

    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cuda",
        compute_type="float16"
    )

else:

    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cpu",
        compute_type="int8"
    )

print(
    "Whisper ready."
)


# ============================================================
# 8. LOAD QWEN SUMMARY MODEL
# ============================================================

print()

print("=" * 70)
print("LOADING SUMMARY MODEL")
print("=" * 70)

print(
    "Model:",
    SUMMARY_MODEL_NAME
)

summary_tokenizer = (
    AutoTokenizer.from_pretrained(
        SUMMARY_MODEL_NAME
    )
)

if DEVICE == "cuda":

    summary_model = (
        AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL_NAME,
            dtype=torch.float16
        )
    )

else:

    summary_model = (
        AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL_NAME,
            dtype=torch.float32
        )
    )

summary_model = summary_model.to(
    DEVICE
)

summary_model.eval()

print(
    "Summary model ready."
)


# ============================================================
# 9. LOAD EMOTION CLASSIFICATION MODEL
# ============================================================

print()

print("=" * 70)
print("LOADING EMOTION CLASSIFICATION MODEL")
print("=" * 70)

print(
    "Model:",
    EMOTION_MODEL_NAME
)

emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    device=0 if DEVICE == "cuda" else -1,
    top_k=None
)

print(
    "Emotion classification model ready."
)


# ============================================================
# 10. FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version="1.0.0"
)


# ============================================================
# 11. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 12. PROGRESS FUNCTION
# ============================================================

def cmd_progress(
    percent,
    message
):

    percent = max(
        0,
        min(
            100,
            int(percent)
        )
    )

    bar_length = 40

    filled = int(
        bar_length * percent / 100
    )

    bar = (
        "#" * filled
        +
        "-" * (
            bar_length - filled
        )
    )

    print(
        f"\r[{bar}] {percent:3d}% | {message}",
        end="",
        flush=True
    )

    if percent >= 100:
        print()


# ============================================================
# 13. TIMESTAMP
# ============================================================

def format_timestamp(
    seconds
):

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:06.3f}"
    )


# ============================================================
# 14. CONVERT AUDIO
# ============================================================

def convert_audio(
    input_audio,
    output_audio
):

    print()

    print(
        "Converting audio to 16 kHz mono..."
    )

    cmd_progress(
        2,
        "Preparing audio"
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output_audio)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            +
            result.stderr
        )

    cmd_progress(
        10,
        "Audio converted"
    )

    return str(output_audio)


# ============================================================
# 15. LOAD WAV AS NUMPY
# ============================================================

def load_wav_numpy(
    wav_file
):

    with wave.open(
        str(wav_file),
        "rb"
    ) as wf:

        channels = wf.getnchannels()

        sample_width = (
            wf.getsampwidth()
        )

        sample_rate = (
            wf.getframerate()
        )

        frame_count = (
            wf.getnframes()
        )

        raw_audio = (
            wf.readframes(
                frame_count
            )
        )

    if sample_width == 2:

        audio = np.frombuffer(
            raw_audio,
            dtype=np.int16
        ).astype(
            np.float32
        )

        audio /= 32768.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw_audio,
            dtype=np.int32
        ).astype(
            np.float32
        )

        audio /= 2147483648.0

    else:

        raise RuntimeError(
            f"Unsupported WAV sample width: "
            f"{sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(
            axis=1
        )

    return (
        audio,
        sample_rate
    )


# ============================================================
# 16. PARSE NEMO RESULT
# ============================================================

def parse_nemo_result(
    raw_result
):

    segments = []

    for item in raw_result:

        text = str(
            item
        ).strip()

        parts = text.split()

        if len(parts) < 3:
            continue

        try:

            start = float(
                parts[0]
            )

            end = float(
                parts[1]
            )

            speaker_text = parts[2]

            match = re.search(
                r"(\d+)",
                speaker_text
            )

            if match:

                speaker_number = int(
                    match.group(1)
                )

            else:

                speaker_number = 0

            speaker = (
                f"Speaker {speaker_number:02d}"
            )

            duration = (
                end - start
            )

            if duration <= 0:
                continue

            segments.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "duration": duration
                }
            )

        except Exception:

            continue

    return sorted(
        segments,
        key=lambda x: x["start"]
    )


# ============================================================
# 17. RUN NEMO
# ============================================================

def run_nemo(
    audio_file
):

    print()

    print(
        "Running NeMo Sortformer..."
    )

    cmd_progress(
        12,
        "Loading audio for NeMo"
    )

    waveform, sample_rate = (
        load_wav_numpy(
            audio_file
        )
    )

    if sample_rate != 16000:

        raise RuntimeError(
            f"Expected 16000 Hz audio, "
            f"got {sample_rate} Hz."
        )

    cmd_progress(
        15,
        "Running speaker diarization"
    )

    with torch.inference_mode():

        result = nemo_model.diarize(
            audio=[waveform],
            sample_rate=16000,
            batch_size=1,
            num_workers=0,
            verbose=True
        )

    cmd_progress(
        45,
        "NeMo diarization completed"
    )

    if not result:

        raise RuntimeError(
            "NeMo returned no result."
        )

    raw_result = result[0]

    segments = parse_nemo_result(
        raw_result
    )

    if not segments:

        raise RuntimeError(
            "NeMo did not produce "
            "speaker segments."
        )

    return segments


# ============================================================
# 18. OVERLAP CALCULATION
# ============================================================

def calculate_overlap(
    start1,
    end1,
    start2,
    end2
):

    overlap_start = max(
        start1,
        start2
    )

    overlap_end = min(
        end1,
        end2
    )

    return max(
        0.0,
        overlap_end - overlap_start
    )


# ============================================================
# 19. FIND SPEAKER
# ============================================================

def find_speaker(
    whisper_start,
    whisper_end,
    speaker_segments
):

    best_speaker = None

    best_overlap = 0.0

    for speaker_segment in speaker_segments:

        overlap = calculate_overlap(
            whisper_start,
            whisper_end,
            speaker_segment["start"],
            speaker_segment["end"]
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = (
                speaker_segment["speaker"]
            )

    if best_speaker is None:

        closest_distance = float("inf")

        for speaker_segment in speaker_segments:

            if whisper_end < speaker_segment["start"]:

                distance = (
                    speaker_segment["start"]
                    -
                    whisper_end
                )

            elif whisper_start > speaker_segment["end"]:

                distance = (
                    whisper_start
                    -
                    speaker_segment["end"]
                )

            else:

                distance = 0

            if distance < closest_distance:

                closest_distance = distance

                best_speaker = (
                    speaker_segment["speaker"]
                )

    return best_speaker


# ============================================================
# 20. TRANSCRIBE COMPLETE AUDIO
# ============================================================

def transcribe_complete_audio(
    audio_file
):

    print()

    print("=" * 70)
    print("TRANSCRIBING COMPLETE AUDIO")
    print("=" * 70)

    cmd_progress(
        50,
        "Whisper transcribing complete audio"
    )

    whisper_segments, info = (
        whisper_model.transcribe(
            audio_file,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True
        )
    )

    results = []

    for segment in whisper_segments:

        text = segment.text.strip()

        if not text:
            continue

        results.append(
            {
                "start": float(
                    segment.start
                ),
                "end": float(
                    segment.end
                ),
                "text": text
            }
        )

    if not results:

        raise RuntimeError(
            "Whisper did not produce "
            "any transcription."
        )

    print(
        "Whisper segments:",
        len(results)
    )

    cmd_progress(
        65,
        "Whisper transcription completed"
    )

    return results


# ============================================================
# 21. ASSIGN SPEAKERS
# ============================================================

def assign_speakers(
    whisper_segments,
    speaker_segments
):

    print()

    print(
        "Matching Whisper timestamps "
        "with NeMo speakers..."
    )

    results = []

    total = len(
        whisper_segments
    )

    for index, whisper_item in enumerate(
        whisper_segments,
        start=1
    ):

        speaker = find_speaker(
            whisper_item["start"],
            whisper_item["end"],
            speaker_segments
        )

        if speaker is None:

            speaker = "Speaker 00"

        results.append(
            {
                "speaker": speaker,
                "start": whisper_item["start"],
                "end": whisper_item["end"],
                "text": whisper_item["text"]
            }
        )

        percent = (
            65
            +
            int(
                10 *
                index /
                max(total, 1)
            )
        )

        cmd_progress(
            percent,
            f"Speaker matching {index}/{total}"
        )

    return results


# ============================================================
# 22. MERGE SAME SPEAKER
# ============================================================

def merge_speaker_segments(
    segments,
    max_gap=1.2
):

    if not segments:
        return []

    merged = []

    current = {
        "speaker": segments[0]["speaker"],
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"].strip()
    }

    for item in segments[1:]:

        same_speaker = (
            item["speaker"]
            ==
            current["speaker"]
        )

        gap = (
            item["start"]
            -
            current["end"]
        )

        if same_speaker and gap <= max_gap:

            current["end"] = (
                item["end"]
            )

            new_text = (
                item["text"].strip()
            )

            if new_text:

                if (
                    current["text"]
                    and
                    not current["text"].endswith(
                        (
                            " ",
                            ".",
                            "!",
                            "?",
                            ",",
                            ";",
                            ":"
                        )
                    )
                ):

                    current["text"] += " "

                current["text"] += new_text

        else:

            merged.append(
                current
            )

            current = {
                "speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "text": item["text"].strip()
            }

    merged.append(
        current
    )

    return merged


# ============================================================
# 23. CLEAN TRANSCRIPT TEXT
# ============================================================

def clean_transcript_text(
    text
):

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    if text and text[-1] not in ".!?":
        text += "."

    return text


# ============================================================
# 24. BUILD TRANSCRIPT
# ============================================================

def build_transcript(
    results
):

    lines = []

    for item in results:

        start = format_timestamp(
            item["start"]
        )

        end = format_timestamp(
            item["end"]
        )

        text = clean_transcript_text(
            item["text"]
        )

        lines.append(
            f"[{start} -> {end}] "
            f"{item['speaker']}: "
            f"{text}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# 25. RUN QWEN
# ============================================================

def run_qwen(
    prompt,
    max_new_tokens
):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a factual meeting assistant. "
                "Never invent information. "
                "Only use information explicitly present "
                "in the transcript."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = (
        summary_tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    inputs = summary_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = (
            summary_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        )

    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return (
        summary_tokenizer.decode(
            generated,
            skip_special_tokens=True
        )
        .strip()
    )


# ============================================================
# 26. EMOTION ANALYSIS HELPERS
# ============================================================

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]


def empty_emotion_scores():

    return {
        label: 0.0
        for label in EMOTION_LABELS
    }


# ============================================================
# 27. ANALYZE SPEAKER EMOTIONS
# ============================================================

def analyze_speaker_emotions(
    merged_transcript
):

    print()

    print("=" * 70)
    print("ANALYZING SPEAKER EMOTIONS")
    print("=" * 70)

    cmd_progress(
        88,
        "Analyzing speaker emotions"
    )

    speaker_data = {}

    # --------------------------------------------------------
    # Group transcript by speaker
    # --------------------------------------------------------

    for item in merged_transcript:

        speaker = item["speaker"]

        text = item["text"].strip()

        if not text:
            continue

        if speaker not in speaker_data:

            speaker_data[speaker] = {
                "texts": [],
                "scores": empty_emotion_scores(),
                "segments": 0
            }

        speaker_data[speaker]["texts"].append(
            text
        )

    # --------------------------------------------------------
    # Analyze each speaker
    # --------------------------------------------------------

    for speaker_index, (
        speaker,
        data
    ) in enumerate(
        speaker_data.items(),
        start=1
    ):

        texts = data["texts"]

        print(
            f"\nAnalyzing {speaker}..."
        )

        for text in texts:

            try:

                # Limit very long text
                text_for_model = text[:1000]

                predictions = (
                    emotion_classifier(
                        text_for_model
                    )
                )

                # Transformers can return:
                #
                # [[
                #   {"label": "...", "score": ...}
                # ]]
                #
                # or:
                #
                # [
                #   {"label": "...", "score": ...}
                # ]

                if (
                    predictions
                    and
                    isinstance(
                        predictions[0],
                        list
                    )
                ):

                    predictions = predictions[0]

                for prediction in predictions:

                    label = (
                        prediction["label"]
                        .lower()
                    )

                    score = float(
                        prediction["score"]
                    )

                    if label in data["scores"]:

                        data["scores"][label] += score

                data["segments"] += 1

            except Exception as e:

                print(
                    f"Emotion warning for {speaker}:",
                    e
                )

        # ----------------------------------------------------
        # Average scores
        # ----------------------------------------------------

        if data["segments"] > 0:

            for label in EMOTION_LABELS:

                data["scores"][label] /= (
                    data["segments"]
                )

        # Normalize to percentages
        total_score = sum(
            data["scores"].values()
        )

        if total_score > 0:

            for label in EMOTION_LABELS:

                data["scores"][label] = (
                    data["scores"][label]
                    /
                    total_score
                    *
                    100
                )

    cmd_progress(
        91,
        "Speaker emotions completed"
    )

    return speaker_data


# ============================================================
# 28. DETERMINE SPEAKER DOMINANT EMOTION
# ============================================================

def dominant_emotion(
    scores
):

    if not scores:
        return "Neutral"

    label = max(
        scores,
        key=scores.get
    )

    names = {
        "anger": "Angry",
        "disgust": "Disgusted",
        "fear": "Fearful",
        "joy": "Positive",
        "neutral": "Neutral",
        "sadness": "Sad",
        "surprise": "Surprised"
    }

    return names.get(
        label,
        label.title()
    )


# ============================================================
# 29. GENERATE EMOTION REPORT
# ============================================================

def generate_emotion_report(
    speaker_data
):

    if not speaker_data:

        return (
            "Meeting Emotions\n\n"
            "No speaker emotion data available."
        )

    # --------------------------------------------------------
    # Overall scores
    # --------------------------------------------------------

    overall_scores = (
        empty_emotion_scores()
    )

    speaker_count = len(
        speaker_data
    )

    for data in speaker_data.values():

        for label in EMOTION_LABELS:

            overall_scores[label] += (
                data["scores"][label]
            )

    for label in EMOTION_LABELS:

        overall_scores[label] /= max(
            speaker_count,
            1
        )

    overall_emotion = dominant_emotion(
        overall_scores
    )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "Meeting Emotions"
    )

    lines.append(
        "────────────────────────"
    )

    lines.append(
        f"Overall: {overall_emotion}"
    )

    lines.append("")

    # --------------------------------------------------------
    # Show overall percentages
    # --------------------------------------------------------

    lines.append(
        "Overall Distribution:"
    )

    sorted_overall = sorted(
        overall_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for label, score in sorted_overall:

        if score < 1:
            continue

        display_name = {
            "anger": "Anger",
            "disgust": "Disgust",
            "fear": "Fear",
            "joy": "Joy",
            "neutral": "Neutral",
            "sadness": "Sadness",
            "surprise": "Surprise"
        }.get(
            label,
            label.title()
        )

        lines.append(
            f"{display_name:<12} "
            f"{score:.0f}%"
        )

    lines.append("")

    # --------------------------------------------------------
    # Per speaker
    # --------------------------------------------------------

    for speaker, data in speaker_data.items():

        lines.append(
            speaker
        )

        sorted_scores = sorted(
            data["scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for label, score in sorted_scores:

            if score < 1:
                continue

            display_name = {
                "anger": "Anger",
                "disgust": "Disgust",
                "fear": "Fear",
                "joy": "Joy",
                "neutral": "Neutral",
                "sadness": "Sadness",
                "surprise": "Surprise"
            }.get(
                label,
                label.title()
            )

            lines.append(
                f"{display_name:<12} "
                f"{score:.0f}%"
            )

        lines.append("")

    return "\n".join(
        lines
    )


# ============================================================
# 30. GENERATE SUMMARY
# ============================================================

def generate_summary(
    transcript
):

    print()

    print(
        "Generating summary..."
    )

    cmd_progress(
        82,
        "Generating summary"
    )

    prompt = f"""
You are a STRICT factual meeting summarizer.

Your job is to summarize ONLY what is explicitly stated
in the transcript.

STRICT RULES:

- Only include information explicitly mentioned in the transcript.
- Do not assume or add common tasks.
- Do not invent action items, people, deadlines, or decisions.
- If something is not mentioned, leave it out.
- Do not guess missing information.
- Paraphrase only confirmed information.
- Keep the Summary within 5 sentences.

Return EXACTLY:

Summary:

Write a maximum of 5 sentences.

TRANSCRIPT:

{transcript}
"""

    result = run_qwen(
        prompt,
        max_new_tokens=400
    )

    cmd_progress(
        86,
        "Summary completed"
    )

    return result


# ============================================================
# 31. GENERATE NOTES
# ============================================================

def generate_notes(
    transcript
):

    print()

    print(
        "Generating notes..."
    )

    cmd_progress(
        87,
        "Generating notes"
    )

    prompt = f"""
Create factual notes from the transcript.

Rules:

- Only include information explicitly mentioned in the transcript.
- Do not assume, guess, or create information.
- Do not invent names, tasks, deadlines, decisions, or meetings.
- Include only confirmed information.
- Each bullet must be a single short sentence or fragment.
- Maximum approximately 12 words per bullet.
- Every bullet MUST start with "- ".
- Do not write paragraphs.
- Do not write long sentences.
- Only produce the two sections below.
- Do not add an introduction.
- Do not add an explanation.

EXACT OUTPUT FORMAT:

Key Points

- confirmed information from transcript

Important Notes

- confirmed information from transcript

TRANSCRIPT:

{transcript}
"""

    result = run_qwen(
        prompt,
        max_new_tokens=350
    )

    cmd_progress(
        92,
        "Notes completed"
    )

    return result


# ============================================================
# 32. SAVE RESULTS
# ============================================================

def save_results(
    job_id,
    original_filename,
    transcript,
    summary,
    notes,
    emotion_report
):

    job_dir = (
        OUTPUT_DIR /
        job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    transcript_file = (
        job_dir /
        "meeting_transcript.txt"
    )

    summary_file = (
        job_dir /
        "meeting_summary_notes.txt"
    )

    email_file = (
        job_dir /
        "meeting_email_draft.txt"
    )

    emotion_file = (
        job_dir /
        "meeting_emotions.txt"
    )

    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    with open(
        transcript_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "MEETING TRANSCRIPT\n"
        )

        f.write(
            "=" * 70 +
            "\n\n"
        )

        f.write(
            f"Audio: {original_filename}\n\n"
        )

        f.write(
            transcript
        )

    # --------------------------------------------------------
    # Summary + Notes
    # --------------------------------------------------------

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            summary
        )

        f.write(
            "\n\n"
        )

        f.write(
            notes
        )

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    email = f"""Subject: Meeting Summary

Hello,

Please find the meeting summary and notes below.

{summary}

{notes}

Regards
"""

    with open(
        email_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            email
        )

    # --------------------------------------------------------
    # Emotion report
    # --------------------------------------------------------

    with open(
        emotion_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            emotion_report
        )

    return (
        transcript_file,
        summary_file,
        email_file,
        emotion_file
    )


# ============================================================
# 33. PROCESS AUDIO
# ============================================================

def process_audio_file(
    input_file
):

    if input_file is None:

        raise RuntimeError(
            "No audio file supplied."
        )

    job_id = uuid.uuid4().hex

    job_dir = (
        OUTPUT_DIR /
        job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    original_name = (
        Path(input_file).name
    )

    input_path = (
        job_dir /
        original_name
    )

    converted_path = (
        job_dir /
        "audio_16k.wav"
    )

    print()

    print("=" * 70)
    print("STARTING AUDIO PROCESSING")
    print("=" * 70)

    cmd_progress(
        0,
        "Starting"
    )

    # --------------------------------------------------------
    # COPY INPUT
    # --------------------------------------------------------

    shutil.copy2(
        input_file,
        input_path
    )

    cmd_progress(
        1,
        "Input audio copied"
    )

    # --------------------------------------------------------
    # CONVERT
    # --------------------------------------------------------

    convert_audio(
        input_path,
        converted_path
    )

    # --------------------------------------------------------
    # NEMO
    # --------------------------------------------------------

    speaker_segments = run_nemo(
        str(converted_path)
    )

    speakers = sorted(
        set(
            item["speaker"]
            for item in speaker_segments
        )
    )

    print()

    print(
        "Speakers detected:",
        len(speakers)
    )

    print(
        "Speakers:",
        ", ".join(speakers)
    )

    # --------------------------------------------------------
    # WHISPER FULL AUDIO
    # --------------------------------------------------------

    whisper_segments = (
        transcribe_complete_audio(
            str(converted_path)
        )
    )

    # --------------------------------------------------------
    # MATCH SPEAKERS
    # --------------------------------------------------------

    speaker_transcript = (
        assign_speakers(
            whisper_segments,
            speaker_segments
        )
    )

    # --------------------------------------------------------
    # MERGE SAME SPEAKER
    # --------------------------------------------------------

    merged_transcript = (
        merge_speaker_segments(
            speaker_transcript,
            max_gap=1.2
        )
    )

    print()

    print(
        "Merged speaker turns:",
        len(merged_transcript)
    )

    # --------------------------------------------------------
    # BUILD TRANSCRIPT
    # --------------------------------------------------------

    transcript = build_transcript(
        merged_transcript
    )

    if not transcript:

        raise RuntimeError(
            "No speech was transcribed."
        )

    cmd_progress(
        80,
        "Transcript completed"
    )

    # --------------------------------------------------------
    # EMOTION ANALYSIS
    # --------------------------------------------------------

    speaker_emotions = (
        analyze_speaker_emotions(
            merged_transcript
        )
    )

    emotion_report = (
        generate_emotion_report(
            speaker_emotions
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = generate_summary(
        transcript
    )

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    notes = generate_notes(
        transcript
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    cmd_progress(
        94,
        "Saving result files"
    )

    (
        transcript_file,
        summary_file,
        email_file,
        emotion_file
    ) = save_results(
        job_id,
        original_name,
        transcript,
        summary,
        notes,
        emotion_report
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    cmd_progress(
        100,
        "Processing completed"
    )

    print()

    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        "Job ID:",
        job_id
    )

    print(
        "Total speakers:",
        len(speakers)
    )

    print(
        "Speaker turns:",
        len(merged_transcript)
    )

    print(
        "Transcript:",
        transcript_file
    )

    print(
        "Summary:",
        summary_file
    )

    print(
        "Email:",
        email_file
    )

    print(
        "Emotions:",
        emotion_file
    )

    print("=" * 70)

    return {

        "job_id": job_id,

        "device": DEVICE,

        "speaker_count": len(speakers),

        "speakers": speakers,

        "transcript": transcript,

        "summary": summary,

        "notes": notes,

        "emotion_report": emotion_report,

        "transcript_file": str(
            transcript_file
        ),

        "summary_file": str(
            summary_file
        ),

        "email_file": str(
            email_file
        ),

        "emotion_file": str(
            emotion_file
        )
    }


# ============================================================
# 34. FASTAPI ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "application": APP_NAME,

        "status": "running",

        "device": DEVICE,

        "cuda": torch.cuda.is_available()
    }


# ============================================================
# 35. FASTAPI PROCESS
# ============================================================

@app.post("/process")
async def process_meeting(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file supplied."
        )

    extension = (
        Path(file.filename).suffix
        or ".audio"
    )

    temp_id = uuid.uuid4().hex

    input_path = (
        TEMP_DIR /
        f"{temp_id}{extension}"
    )

    try:

        content = await file.read()

        with open(
            input_path,
            "wb"
        ) as f:

            f.write(
                content
            )

        result = process_audio_file(
            str(input_path)
        )

        return result

    except Exception as e:

        print()

        print(
            "PROCESS ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        try:

            if input_path.exists():

                try:

                    input_path.unlink()

                except PermissionError:

                    print(
                        "Temporary upload is still "
                        "locked by Windows."
                    )

        except Exception as e:

            print(
                "Cleanup warning:",
                e
            )


# ============================================================
# 36. DOWNLOAD TRANSCRIPT
# ============================================================

@app.get(
    "/download/{job_id}/transcript"
)
def download_transcript(
    job_id: str
):

    path = (
        OUTPUT_DIR /
        job_id /
        "meeting_transcript.txt"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Transcript not found."
        )

    return FileResponse(
        path,
        filename="meeting_transcript.txt"
    )


# ============================================================
# 37. DOWNLOAD SUMMARY
# ============================================================

@app.get(
    "/download/{job_id}/summary"
)
def download_summary(
    job_id: str
):

    path = (
        OUTPUT_DIR /
        job_id /
        "meeting_summary_notes.txt"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Summary not found."
        )

    return FileResponse(
        path,
        filename="meeting_summary_notes.txt"
    )


# ============================================================
# 38. DOWNLOAD EMAIL
# ============================================================

@app.get(
    "/download/{job_id}/email"
)
def download_email(
    job_id: str
):

    path = (
        OUTPUT_DIR /
        job_id /
        "meeting_email_draft.txt"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Email draft not found."
        )

    return FileResponse(
        path,
        filename="meeting_email_draft.txt"
    )


# ============================================================
# 39. DOWNLOAD EMOTIONS
# ============================================================

@app.get(
    "/download/{job_id}/emotions"
)
def download_emotions(
    job_id: str
):

    path = (
        OUTPUT_DIR /
        job_id /
        "meeting_emotions.txt"
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Emotion report not found."
        )

    return FileResponse(
        path,
        filename="meeting_emotions.txt"
    )


# ============================================================
# 40. GRADIO PROCESS
# ============================================================

def gradio_process(
    audio_file
):

    if audio_file is None:

        return (
            "Please record or upload audio.",
            "",
            "",
            "",
            "",
            "",
            None,
            None
        )

    try:

        result = process_audio_file(
            audio_file
        )

        # ----------------------------------------------------
        # CREATE EMAIL FOR DISPLAY
        # ----------------------------------------------------

        email_text = f"""Subject: Meeting Summary

Hello,

Please find the meeting summary and notes below.

{result["summary"]}

{result["notes"]}

Regards
"""

        return (

            f"""Completed successfully.

Device: {result["device"]}

Speakers detected: {result["speaker_count"]}

Speakers:

{", ".join(result["speakers"])}
""",

            result["transcript"],

            result["summary"],

            result["notes"],

            result["emotion_report"],

            email_text,

            result["transcript_file"],

            result["summary_file"]
        )

    except Exception as e:

        print(
            "GRADIO ERROR:",
            repr(e)
        )

        return (

            "ERROR: " + str(e),

            "",

            "",

            "",

            "",

            "",

            None,

            None
        )


# ============================================================
# 41. GRADIO CSS
# ============================================================

css = """

body {
    background: white;
}

.header {
    text-align: center;
    padding: 30px;
    border-bottom: 1px solid #e5e7eb;
}

.logo {
    color: #2596be;
    font-size: 55px;
    font-weight: 700;
    font-style: italic;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 20px;
    margin-top: 50px;
    margin-bottom: 35px;
}

.process-button button {
    background: #2596be !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 15px 40px !important;
    font-size: 18px !important;
}

.footer {
    text-align: center;
    color: #9ca3af;
    padding: 20px;
    margin-top: 30px;
}

.output-tabs {
    min-height: 500px;
}

.status-box {
    margin-top: 15px;
}

.emotion-box textarea {
    font-family: monospace !important;
}

"""


# ============================================================
# 42. GRADIO APP
# ============================================================

with gr.Blocks(
    title="AI Meeting Assistant",
    css=css,
    theme=gr.themes.Soft()
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="header">

            <div class="logo">
                AI Meeting Assistant
            </div>

        </div>
        """
    )


    # ========================================================
    # SUBTITLE
    # ========================================================

    gr.HTML(
        """
        <div class="subtitle">

            Convert meetings into speaker transcripts,
            summaries, notes, emotions and email drafts
            using AI

        </div>
        """
    )


    # ========================================================
    # MAIN CONTENT
    # ========================================================

    with gr.Row():

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column(
            scale=1
        ):

            audio = gr.Audio(
                sources=[
                    "microphone",
                    "upload"
                ],
                type="filepath",
                label="🎙️ Record or Upload Meeting Audio",
                interactive=True
            )

            process_button = gr.Button(
                "🚀 Process Meeting",
                elem_classes="process-button",
                variant="primary"
            )

            status = gr.Textbox(
                label="Status",
                lines=8,
                interactive=False,
                elem_classes="status-box"
            )


        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column(
            scale=2
        ):

            with gr.Tabs(
                elem_classes="output-tabs"
            ):

                # ============================================
                # TRANSCRIPT
                # ============================================

                with gr.Tab(
                    "📝 Transcript"
                ):

                    transcript = gr.Textbox(
                        label="Speaker Transcript",
                        lines=25,
                        interactive=False,
                        placeholder=(
                            "Speaker transcript will appear here..."
                        )
                    )


                # ============================================
                # SUMMARY
                # ============================================

                with gr.Tab(
                    "📋 Summary"
                ):

                    summary = gr.Textbox(
                        label="Meeting Summary",
                        lines=15,
                        interactive=False,
                        placeholder=(
                            "Meeting summary will appear here..."
                        )
                    )


                # ============================================
                # NOTES
                # ============================================

                with gr.Tab(
                    "🗒️ Notes"
                ):

                    notes = gr.Textbox(
                        label="Meeting Notes",
                        lines=20,
                        interactive=False,
                        placeholder=(
                            "Meeting notes will appear here..."
                        )
                    )


                # ============================================
                # EMOTIONS
                # ============================================

                with gr.Tab(
                    "🎭 Emotions"
                ):

                    emotion_output = gr.Textbox(
                        label="Speaker Emotion Analysis",
                        lines=30,
                        interactive=False,
                        placeholder=(
                            "Speaker emotions will appear here..."
                        ),
                        elem_classes="emotion-box"
                    )


                # ============================================
                # EMAIL
                # ============================================

                with gr.Tab(
                    "📧 Email"
                ):

                    email_output = gr.Textbox(
                        label="Generated Meeting Email",
                        lines=18,
                        interactive=False,
                        placeholder=(
                            "Generated email draft will appear here..."
                        )
                    )

                    recipient_email = gr.Textbox(
                        label="Recipient Email",
                        placeholder=(
                            "Enter recipient email address..."
                        ),
                        interactive=True
                    )


                # ============================================
                # DOWNLOAD
                # ============================================

                with gr.Tab(
                    "⬇️ Download"
                ):

                    gr.Markdown(
                        """
                        ### 📥 Download Meeting Results

                        Download the generated transcript,
                        summary/notes and emotion report
                        after processing.
                        """
                    )

                    transcript_file = gr.File(
                        label="📝 Download Transcript",
                        interactive=False
                    )

                    summary_file = gr.File(
                        label="📋 Download Summary + Notes",
                        interactive=False
                    )


    # ========================================================
    # PROCESS EVENT
    # ========================================================

    process_button.click(

        fn=gradio_process,

        inputs=[
            audio
        ],

        outputs=[
            status,
            transcript,
            summary,
            notes,
            emotion_output,
            email_output,
            transcript_file,
            summary_file
        ]
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            <p>
                © 2026 AI Meeting Assistant
            </p>

            <p style="font-size: 12px;">

                Powered by NeMo Sortformer +
                Faster-Whisper + Qwen +
                Emotion Classification

            </p>

            <p style="font-size: 12px;">

                Speaker Diarization • Transcription •
                Emotion Analysis • Summary • Notes

            </p>

        </div>
        """
    )


# ============================================================
# 43. START GRADIO
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "STARTING GRADIO"
    )

    print("=" * 70)

    print(
        "Public Gradio link will appear below."
    )

    demo.launch(
        share=True,
        debug=True
    )