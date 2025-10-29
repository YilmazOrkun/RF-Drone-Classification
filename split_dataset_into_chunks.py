import torch, numpy as np, os, math

# --- Load full dataset ---
data = torch.load("dataset.pt", map_location="cpu")
X = data["x_spec"]     # shape [98705, 2, 128, 128]
y = data["y"]
snr = data["snr"]
duty = data["duty_cycle"]

N = len(y)
print(f"Loaded dataset: {N} samples")

# --- Global shuffle ---
idx = torch.randperm(N)
X = X[idx]
y = y[idx]
snr = snr[idx]
duty = duty[idx]

# --- Estimate chunk size ---
samples_per_chunk = 10000   # ≈ 1 GB per chunk for 128×128×2 float32
os.makedirs("chunks_mixed", exist_ok=True)
n_chunks = math.ceil(N / samples_per_chunk)

for i in range(n_chunks):
    start, end = i * samples_per_chunk, min((i+1) * samples_per_chunk, N)
    np.savez_compressed(
        f"chunks_mixed/chunk_{i:02d}.npz",
        x_spec=X[start:end].numpy(),
        y=y[start:end].numpy(),
        snr=snr[start:end].numpy(),
        duty_cycle=duty[start:end].numpy(),
    )
    print(f"✅ Saved chunks_mixed/chunk_{i:02d}.npz ({end-start} samples)")

