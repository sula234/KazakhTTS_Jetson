# grpc_server/server.py
import io, os
import grpc
from concurrent import futures

from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
import torch, soundfile as sf

from grpc_server import tts_pb2, tts_pb2_grpc

# ---- Model paths (mounted under /workspace/models) ----
TTS_CONFIG  = os.environ.get("TTS_CONFIG",  "/workspace/models/config.yaml")
TTS_MODEL   = os.environ.get("TTS_MODEL",   "/workspace/models/train.loss.ave_5best.pth")
VOCODER_CKPT= os.environ.get("VOCODER_CKPT","/workspace/models/vocoders/checkpoint-400000steps.pkl")
DEFAULT_SR  = int(os.environ.get("DEFAULT_SR", "22050"))

# ---- Load once at startup ----
print("Loading TTS...")
text2speech = Text2Speech(
    TTS_CONFIG, TTS_MODEL, device="cuda",
    threshold=0.5, minlenratio=0.0, maxlenratio=10.0,
    use_att_constraint=True, backward_window=1, forward_window=3,
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None  # disable griffin-lim

print("Loading vocoder...")
vocoder = load_model(VOCODER_CKPT).to("cuda").eval()
vocoder.remove_weight_norm()
print("Models ready.")

class TTSService(tts_pb2_grpc.TTSServicer):
    def Synthesize(self, request, context):
        text = (request.text or "").strip()
        if not text:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text is empty")

        sr = request.sample_rate or DEFAULT_SR
        with torch.no_grad():
            out = text2speech(text.lower())
            feat_gen = out["feat_gen"]
            wav = vocoder.inference(feat_gen).view(-1).cpu().numpy()

        # Encode as WAV in-memory (PCM16)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return tts_pb2.AudioResponse(wav=buf.getvalue(), sample_rate=sr, note="ok")

def serve():
    opts = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=opts)
    tts_pb2_grpc.add_TTSServicer_to_server(TTSService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC TTS server listening on :50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
