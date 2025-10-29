import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import RFSpecCNN
import glob, gc, collections, os

# ---------------- SETTINGS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 8
train_ratio = 0.8
batch_size = 32
learning_rate = 1e-4
patience = 2
data_dir = "chunks_mixed"     # your new mixed dataset folder

print(f"🧠 Training on {device}")

# ---------------- LOAD FILES ----------------
files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
if not files:
    raise FileNotFoundError(f"No .npz chunks found in {data_dir}/")

# ---------- CLASS COUNTS ----------
counts = collections.Counter()
for f in files:
    y = np.load(f, mmap_mode="r")["y"]
    counts.update(y.tolist())
n_classes = max(counts.keys()) + 1
print(f"🧩 Detected {n_classes} total classes.")
print("📊 Class counts:", dict(counts))

# ---------- CLASS WEIGHTS ----------
freq = np.array([counts[i] for i in range(n_classes)], dtype=np.float32)
weights = 1.0 / (freq + 1e-6)
weights = weights / weights.sum() * len(freq)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("⚖️  Class weights:", class_weights)

# ---------- GLOBAL NORMALIZATION ----------
print("🧮 Estimating global mean/std from first chunk ...")
d = np.load(files[0], mmap_mode="r")
Xtmp = d["x_spec"]
mean = Xtmp.mean(axis=(0, 2, 3))
std = Xtmp.std(axis=(0, 2, 3)) + 1e-8
d.close()
print(f"Mean={mean}, Std={std}")

# ---------------- MODEL ----------------
model = RFSpecCNN(n_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val = 0
no_improve = 0

# ---------------- TRAINING LOOP ----------------
for epoch in range(1, num_epochs + 1):
    print(f"\n🌀 Epoch {epoch}/{num_epochs}")
    np.random.shuffle(files)
    model.train()
    epoch_loss = 0.0

    for f in files:
        data = np.load(f, mmap_mode="r")
        X = torch.from_numpy(data["x_spec"]).float()
        y = torch.from_numpy(data["y"]).long()
        data.close()

        # normalize
        for c in range(X.shape[1]):
            X[:, c] = (X[:, c] - mean[c]) / std[c]

        # Split into train / validation
        N = len(X)
        train_size = int(train_ratio * N)
        val_size = N - train_size
        train_set, val_set = random_split(TensorDataset(X, y), [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

        # ---- Train ----
        total_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Training on {os.path.basename(f)}", leave=False):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        epoch_loss += avg_loss

        # ---- Validate ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"✅ {os.path.basename(f)} done: loss={avg_loss:.4f}, val_acc={val_acc:.3f}")

        # cleanup
        del X, y, train_set, val_set, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    avg_epoch_loss = epoch_loss / len(files)
    print(f"📉 Epoch {epoch}: mean_loss={avg_epoch_loss:.4f}")

    # ---- Early stopping ----
    if val_acc > best_val + 0.002:
        best_val = val_acc
        no_improve = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_classes": n_classes,
            "class_labels": list(range(n_classes)),
            "class_weights": class_weights.cpu().tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }, "rf_classifier_mixed.pt")
        print("💾 Saved checkpoint (new best).")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⏹️ Early stopping — no improvement.")
            break

print("\n🎯 Training complete!")
print(f"💾 Best model saved as rf_classifier_mixed.pt (val_acc={best_val:.3f})")

