import torch
import numpy as np
import random

# --- SETTINGS ---
dataset_path = "dataset.pt"
output_file = "dataset_replay.c64"
n_samples = 1000  # number of random IQ samples to export

print(f"📂 Loading {dataset_path} ...")
data = torch.load(dataset_path, map_location="cpu")
X_iq = data["x_iq"]  # shape [N, 2, 16384] (I,Q)
N = len(X_iq)

print(f"✅ Loaded dataset: {N} total IQ samples")

# --- RANDOMLY PICK n_samples ---
indices = random.sample(range(N), min(n_samples, N))
print(f"🎲 Selected {len(indices)} random samples")

# --- WRITE OUT AS CONTINUOUS complex64 STREAM ---
with open(output_file, "wb") as f:
    for i in indices:
        I = X_iq[i, 0].numpy().astype(np.float32)
        Q = X_iq[i, 1].numpy().astype(np.float32)
        iq = I + 1j * Q
        iq.astype(np.complex64).tofile(f)

print(f"💾 Wrote {len(indices)} random samples to {output_file}")

