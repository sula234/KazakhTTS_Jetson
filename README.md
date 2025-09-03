# Kazakh\_TTS on Jetson

This project is a **Jetson-ready version** of [IS2AI/Kazakh\_TTS](https://github.com/IS2AI/Kazakh_TTS).
It lets you run **Kazakh text-to-speech (TTS)** models on NVIDIA Jetson devices (Nano, Xavier, Orin) with GPU acceleration, plus a **gRPC interface** for easy integration.

## Setup

### 1. Download pretrained models

Download an acoustic model + vocoder from [IS2AI/Kazakh\_TTS](https://github.com/IS2AI/Kazakh_TTS) and place them in `models/`.

Example structure:

```
models/
  female1/
    config.yaml
    train.loss.ave_5best.pth
  vocoders/female1/
    checkpoint-400000steps.pkl
```

### 3. Run with Docker

```bash
docker compose up --build
```

This starts the gRPC server on port `50052`.

---

## Usage

### Client example

```bash
python3 grpc_server/client_example.py --text "Сәлеметсіз бе, әлем!" --out out.wav
```

This saves the synthesized speech to `out.wav`.
