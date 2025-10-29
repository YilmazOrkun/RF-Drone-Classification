import zmq
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from model import RFSpecCNN

# ---------------- SETTINGS ----------------
ENDPOINT = "tcp://127.0.0.1:5556"   # must match ZMQ PUB Sink in GRC
FFT_SIZE = 128
N_FRAMES = 128
PRINT_INTERVAL = 1.0                # seconds between console prints
PLOT_INTERVAL = 0.1                 # seconds between plot updates

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Live inference on {device}")

# ---------------- LOAD MODEL ----------------
ckpt = torch.load("rf_classifier_mixed.pt", map_location=device, weights_only=False)
n_classes = ckpt["n_classes"]
mean = np.array(ckpt["mean"])
std = np.array(ckpt["std"])
model = RFSpecCNN(n_classes).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"✅ Loaded model with {n_classes} classes")

# ---------------- CLASS LABELS ----------------
CLASS_NAMES = {
    0: "DJI",
    1: "FutabaT14",
    2: "FutabaT7",
    3: "Graupner",
    4: "Noise",
    5: "Taranis",
    6: "Turnigy",
}

# ---------------- CONNECT TO GNU RADIO ----------------
ctx = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect(ENDPOINT)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.setsockopt(zmq.RCVHWM, 1000)
print(f"🔌 Connected to GNU Radio stream at {ENDPOINT}")

# ---------------- BUFFERS ----------------
buf_ch0 = np.zeros((N_FRAMES, FFT_SIZE), dtype=np.float32)
buf_ch1 = np.zeros((N_FRAMES, FFT_SIZE), dtype=np.float32)
ptr = 0
filled = False
last_print = time.time()
last_plot = time.time()

def vector_to_2ch(v: np.ndarray):
    """Convert complex FFT vector -> two float channels."""
    c0 = np.abs(np.real(v)).astype(np.float32)
    c1 = np.abs(np.imag(v)).astype(np.float32)
    return c0, c1

def normalize(spec):
    out = spec.copy()
    for c in range(2):
        out[c] = (out[c] - mean[c]) / (std[c] + 1e-8)
    return out

# ---------------- SETUP LIVE PLOT ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))
img = ax.imshow(
    np.zeros((N_FRAMES, FFT_SIZE)),
    origin="lower",
    aspect="auto",
    cmap="viridis",
    interpolation="nearest"
)
ax.set_title("Live Spectrogram")
ax.set_xlabel("Frequency bins")
ax.set_ylabel("Time frames")
plt.tight_layout()
plt.show(block=False)

# ---------------- MAIN LOOP ----------------
print("🏁 Waiting for FFT frames from GNU Radio...")
while True:
    try:
        msg = socket.recv(flags=0)
        v = np.frombuffer(msg, dtype=np.complex64)
        if v.size != FFT_SIZE:
            continue  # skip malformed messages

        c0, c1 = vector_to_2ch(v)
        buf_ch0[ptr, :] = c0
        buf_ch1[ptr, :] = c1
        ptr = (ptr + 1) % N_FRAMES
        if ptr == 0:
            filled = True

        if not filled:
            continue

        # Build (2,128,128) tensor
        spec = np.stack([buf_ch0, buf_ch1], axis=0)
        spec = normalize(spec)
        X = torch.from_numpy(spec).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            out = model(X)
            pred = out.argmax(1).item()
            label = CLASS_NAMES.get(pred, f"Unknown({pred})")

        now = time.time()

        # Console output
        if now - last_print >= PRINT_INTERVAL:
            print(f"🎯 Predicted class: {label}")
            last_print = now

        # Plot update
        if now - last_plot >= PLOT_INTERVAL:
            img.set_data(buf_ch0)
            img.autoscale()
            ax.set_title(f"Live Spectrogram | Pred: {label}")
            plt.pause(0.001)
            last_plot = now

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(0.1)

