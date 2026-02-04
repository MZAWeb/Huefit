# Hue

Capture video from your webcam, detect your shirt color using OpenAI's GPT-5.2 vision API, and automatically set your Philips Hue lights to match — at maximum brightness.

## Prerequisites

- Python 3.14+
- A Philips Hue Bridge on the same network
- An OpenAI API key with access to GPT-5.2

## Setup

```bash
uv sync
```

On first run, the script will prompt you for your Hue Bridge IP and OpenAI API key. You may also need to press the link button on your Hue Bridge.

Set `TARGET_HUE_ROOM_NAME` in `hue.py` to the name of the Hue room you want to control (defaults to "Office").

## Usage

```bash
uv run hue.py
```

- Press **c** to capture your shirt color and update the lights
- Press **q** to quit
