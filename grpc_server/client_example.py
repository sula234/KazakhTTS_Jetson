#!/usr/bin/env python3
import grpc
import wave
import numpy as np
import sounddevice as sd

import tts_pb2
import tts_pb2_grpc


def request_generator(text: str, sample_rate: int):
    """Send one text request (you could yield multiple if needed)."""
    yield tts_pb2.SynthesisRequest(text=text, sample_rate=sample_rate)


def main():
    server = "localhost:50052"   # if running on host; use "tts:50051" inside docker-compose
    text = "Сәлем әлем! Бұл gRPC тест."
    sample_rate = 22050
    out_file = "out.wav"

    # Connect (plaintext)
    channel = grpc.insecure_channel(server)
    stub = tts_pb2_grpc.TTSStub(channel)

    # Call the streaming RPC
    responses = stub.Synthesize(request_generator(text, sample_rate))

    # Collect PCM bytes
    pcm_bytes = bytearray()
    for chunk in responses:
        pcm_bytes.extend(chunk.pcm16le)
        if chunk.end_of_stream:
            break

    # Save to WAV (16-bit PCM mono)
    with wave.open(out_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    print(f"Saved synthesized speech to {out_file}")

    # Try to play it (needs audio passthrough working in Docker!)
    try:
        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        sd.play(audio, sample_rate)
        sd.wait()
        print("Playback done.")
    except Exception as e:
        print(f"(Skipping playback) {e}")


if __name__ == "__main__":
    main()
