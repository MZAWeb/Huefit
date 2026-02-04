# Hue

Capture video from your webcam, detect your shirt color using OpenAI's GPT-5.2 vision API, and automatically set your Philips Hue lights to match.

## Prerequisites

- Python 3.14+
- A Philips Hue Bridge on the same network
- An OpenAI API key with access to GPT-5.2

## Setup

```bash
uv sync
cp hue.conf.sample hue.conf
```

Edit `hue.conf` and fill in your `bridge_ip` and `api_key`. You can also change the `room`, `brightness` (0-100%), and `model` settings.

On first run, if `bridge_username` is empty, the script will prompt you to press the link button on your Hue Bridge and will save the username automatically.

## Usage

```bash
uv run hue.py
```

- Press **c** to capture your shirt color and update the lights
- Press **q** to quit
