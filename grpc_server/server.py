from concurrent import futures
import grpc, numpy as np
from . import tts_pb2, tts_pb2_grpc

# ---- Your real ESPnet pipeline (load once at import/start) ----
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
import torch, os

# ---- Model paths (mounted under /workspace/models) ----
TTS_CONFIG  = os.environ.get("TTS_CONFIG",  "/workspace/models/config.yaml")
TTS_MODEL   = os.environ.get("TTS_MODEL",   "/workspace/models/train.loss.ave_5best.pth")
VOCODER_CKPT= os.environ.get("VOCODER_CKPT","/workspace/models/vocoders/checkpoint-400000steps.pkl")
DEFAULT_SR  = int(os.environ.get("DEFAULT_SR", "22050"))

print("Loading ESPnet TTS…")
tts = Text2Speech(
    TTS_CONFIG, TTS_MODEL, device="cuda",
    threshold=0.5, minlenratio=0.0, maxlenratio=10.0,
    use_att_constraint=True, backward_window=1, forward_window=3,
    speed_control_alpha=1.0,
)
tts.spc2wav = None

print("Loading PWG vocoder…")
vocoder = load_model(VOCODER_CKPT).to("cuda").eval()
vocoder.remove_weight_norm()
print("Ready.")

# Helper: slice a full waveform into fixed-size frames (e.g. 20 ms)
def slice_frames(wav_np: np.ndarray, sr: int, ms: float = 20.0):
    n = int(sr * (ms / 1000.0))  # samples per frame
    n -= n % 2  # even number of samples (since 16-bit samples are 2 bytes)
    for i in range(0, len(wav_np), n):
        chunk = wav_np[i:i+n]
        if len(chunk) == 0:
            break
        # to int16 little-endian
        pcm16 = np.clip(chunk * 32768.0, -32768, 32767).astype("<i2").tobytes() \
                if chunk.dtype != np.int16 else chunk.tobytes()
        yield pcm16

class TTSService(tts_pb2_grpc.TTSServicer):
    def Synthesize(self, request_iter, context):
        # read first (and only) request
        req = next(request_iter)
        text = (req.text or "").strip()
        if not text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("text is empty")
            return iter([])
        sr = req.sample_rate or DEFAULT_SR

        # run full inference (you can also make this incremental if your model supports it)
        with torch.no_grad():
            out = tts(text.lower())
            feat_gen = out["feat_gen"]
            wav = vocoder.inference(feat_gen).view(-1).cpu().numpy()

        # stream frames as soon as they’re sliced
        for pcm in slice_frames(wav, sr, ms=20.0):
            yield tts_pb2.AudioChunk(pcm16le=pcm, sample_rate=sr)

        # tell client we're done
        yield tts_pb2.AudioChunk(end_of_stream=True, sample_rate=sr)

def serve():
    opts = [
        ("grpc.max_send_message_length", 20*1024*1024),
        ("grpc.max_receive_message_length", 20*1024*1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=opts)
    tts_pb2_grpc.add_TTSServicer_to_server(TTSService(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    print("gRPC streaming TTS on :50052")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()