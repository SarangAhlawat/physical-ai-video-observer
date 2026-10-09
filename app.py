import base64
import os
import tempfile
from pathlib import Path

import cv2
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

st.set_page_config(page_title="Physical AI Video Observer", page_icon="🎥", layout="wide")
st.title("Physical AI Video Observer")
st.caption("A small video-understanding prototype using an NVIDIA-hosted vision-language model.")
st.warning(
    "Prototype only. This is not a certified safety system. Model outputs can be wrong; "
    "never use them alone for real-world safety decisions."
)

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    st.info("Add your NVIDIA API key to a local .env file (see README.md), then restart the app.")

uploaded = st.file_uploader("Upload a short video (MP4, MOV, or AVI)", type=["mp4", "mov", "avi"])
question = st.text_area(
    "What should the model inspect?",
    value=(
        "Analyze the sampled video frames as a Physical AI perception task.\n\n"
        "Return these sections:\n\n"
        "1. SCENE OVERVIEW\n"
        "Describe the environment, visible people, objects, and relevant spatial relationships. "
        "Avoid unnecessary appearance details.\n"
        "2. OBSERVED ACTIONS\n"
        "Describe only actions directly supported by the sampled frames. Distinguish observed "
        "actions from assumptions.\n"
        "3. TEMPORAL CHANGES\n"
        "Compare the frames chronologically. Identify what changed, what appears unchanged, and "
        "what cannot be determined because frames between samples were not observed.\n"
        "4. PHYSICAL INTERACTION\n"
        "Identify visible interactions between people, objects, and the environment. Do not infer "
        "contact or intent without evidence.\n"
        "5. POTENTIAL OPERATIONAL CONCERNS\n"
        "Identify only concerns supported by visible evidence. Explain the evidence and uncertainty. "
        "If none are apparent in the sampled frames, say so without claiming the entire video is safe.\n"
        "6. EVIDENCE TABLE\n"
        "For each important finding, provide the sampled timestamp, observation, confidence (high, "
        "medium, or low), and reason.\n"
        "7. PERCEPTION LIMITATIONS\n"
        "Explain which events may have been missed due to sparse frame sampling, occlusion, poor "
        "visibility, or ambiguity.\n"
        "8. NEXT REVIEW STEP\n"
        "Suggest the most useful additional observation or human review.\n\n"
        "Rules:\n\n"
        "- Do not infer a person's intentions, destination, emotions, or identity.\n"
        "- Do not invent events between sampled frames.\n"
        "- Treat timestamps as approximate sample times, not exact event boundaries.\n"
        "- Distinguish observations from interpretations.\n"
        "- This is an experimental perception assistant, not a safety-certified system."
    ),
    height=360,
)
sample_count = st.slider("Frames to sample", min_value=2, max_value=8, value=4)
max_tokens = st.slider("Maximum response length", min_value=200, max_value=900, value=500)

def extract_frames(video_path: str, count: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open this video. Try a standard MP4 file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        cap.release()
        raise ValueError("The video contains no readable frames.")
    indices = sorted(set(round(i * (total - 1) / max(count - 1, 1)) for i in range(count)))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # Resize to keep requests manageable.
        h, w = frame.shape[:2]
        scale = min(1.0, 960 / max(w, h))
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            timestamp = idx / fps if fps > 0 else float(idx)
            data = base64.b64encode(encoded.tobytes()).decode("utf-8")
            frames.append((timestamp, data, frame))
    cap.release()
    if not frames:
        raise ValueError("Could not extract frames from this video.")
    return frames

def request_model(content, max_tokens):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a careful video-analysis assistant for prototyping. Distinguish observations from guesses."},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(
            f"API request failed ({response.status_code}). "
            f"The provider returned: {response.text[:3000]}"
        )
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"The provider returned an unexpected response: {response.text[:3000]}") from exc

if uploaded:
    st.video(uploaded)
    if st.button("Analyze video", type="primary", disabled=not bool(api_key)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            video_path = tmp.name
        try:
            with st.spinner("Sampling frames and asking the model..."):
                frames = extract_frames(video_path, sample_count)
                st.caption(
                    f"Sampled {len(frames)} frames from {uploaded.name}. "
                    "Each frame is checked separately before the final summary."
                )
                observations = []
                progress = st.progress(0, text="Analyzing sampled frames...")
                for index, (timestamp, image_b64, _) in enumerate(frames, start=1):
                    progress.progress(
                        (index - 1) / len(frames),
                        text=f"Analyzing frame {index} of {len(frames)} ({timestamp:.2f}s)...",
                    )
                    observations.append(
                        request_model([
                            {"type": "text", "text": (
                                f"This is one sampled video frame at {timestamp:.2f} seconds. "
                                f"User task: {question}\n\n"
                                "Describe only what is directly visible in this frame and clearly label any uncertainty."
                            )},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ], max_tokens)
                    )
                progress.progress(1.0, text="Synthesizing the timestamped observations...")
                synthesis = "\n\n".join(
                    f"Frame at {timestamp:.2f} seconds:\n{observation}"
                    for (timestamp, _, _), observation in zip(frames, observations)
                )
                result = request_model(
                    "Synthesize the following timestamped frame observations for this task: "
                    f"{question}\n\n{synthesis}\n\n"
                    "Return: (1) concise scene summary, (2) timeline with approximate timestamps, "
                    "(3) directly visible evidence, (4) uncertain interpretations/limitations, "
                    "(5) suggested next review step. Do not invent events between sampled frames.",
                    max_tokens,
                )
                st.subheader("Analysis")
                st.markdown(result)
                st.download_button(
                    "Download analysis as Markdown",
                    data=f"# Physical AI Video Observer — Analysis\n\n**Video:** {uploaded.name}\n\n**Model:** {MODEL}\n\n{result}\n",
                    file_name="video_analysis.md",
                    mime="text/markdown",
                )
                st.subheader("Sampled frames")
                cols = st.columns(min(4, len(frames)))
                for i, (timestamp, _, frame) in enumerate(frames):
                    with cols[i % len(cols)]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{timestamp:.2f}s")
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Network/API error: {exc}")
        except Exception as exc:
            st.error(f"Could not analyze video: {exc}")
        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass

st.divider()
st.caption("Built as a learning POC. Sampled frames are sent to the configured model API when you click Analyze.")
