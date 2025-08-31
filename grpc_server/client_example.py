# client/client.py
import grpc, wave
from . import tts_pb2, tts_pb2_grpc

with grpc.insecure_channel("localhost:50052") as ch:
    stub = tts_pb2_grpc.TTSStub(ch)
    resp = stub.Synthesize(tts_pb2.SynthesisRequest(text="Сәлем Султан", sample_rate=22050))
    with open("out.wav", "wb") as f:
        f.write(resp.wav)
print("Wrote out.wav")
