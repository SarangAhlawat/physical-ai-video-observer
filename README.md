# Physical AI Video Observer

Physical AI Video Observer analyzes short video clips with a configurable vision-language model. It samples representative frames, describes visible activity, compares observations over time, highlights evidence and uncertainty, and exports the result as Markdown.

The application is designed for exploratory video review and human-in-the-loop analysis. It does not control equipment, make autonomous decisions, or provide safety certification.

## Features

- Upload MP4, MOV, or AVI video clips.
- Sample between 2 and 8 frames across the clip.
- Analyze each frame independently to respect provider image limits.
- Produce a timestamped summary of observations, changes, interactions, concerns, and limitations.
- Display sampled frames alongside the analysis.
- Download the result as a Markdown report.

## Requirements

- Python 3.10 or newer
- An API key for a compatible OpenAI-style vision-language endpoint
- A short, non-sensitive video clip for analysis

## Installation

Create an isolated environment and install the application dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy the example configuration and set credentials locally:

```bash
cp .env.example .env
```

Configure these variables in `.env`:

| Variable | Description |
| --- | --- |
| `NVIDIA_API_KEY` | API key for the configured model provider. |
| `NVIDIA_MODEL` | Vision-language model identifier. |
| `NVIDIA_API_URL` | OpenAI-compatible chat completions endpoint. |

The `.env` file is local configuration and must not be committed or shared.

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a video, review the sampled frames, and select **Analyze video**.

## Processing flow

```text
Video upload -> frame sampling -> per-frame vision analysis -> timestamped synthesis -> human review -> Markdown export
```

The application sends sampled frames to the configured provider only after analysis is requested. The number of API calls is approximately the number of sampled frames plus one synthesis request.

## Limitations and responsible use

- Frame sampling can miss events that occur between sampled timestamps.
- Occlusion, image quality, camera movement, and ambiguous scenes can reduce accuracy.
- Timestamps indicate approximate sample positions, not exact event boundaries.
- Model responses may be incomplete, incorrect, or overconfident.
- Do not use the output as the sole basis for safety, security, medical, legal, or operational decisions.
- Do not upload confidential, personal, or sensitive footage unless the applicable provider and organization policies permit it.
- Review the provider's current model availability, retention, privacy, and usage policies before deployment.

## Operational notes

- Keep API credentials in a secret manager or protected environment variables outside local testing.
- Set a provider budget and request limit before sharing the application with other users.
- Validate model behavior with representative clips and human-reviewed ground truth before relying on results.
- Pin dependency versions for repeatable deployments and monitor provider API changes.

## Project structure

- `app.py`: Streamlit application and video-analysis workflow.
- `requirements.txt`: Python dependencies.
- `.env.example`: configuration template without credentials.
- `.streamlit/config.toml`: Streamlit runtime configuration.
