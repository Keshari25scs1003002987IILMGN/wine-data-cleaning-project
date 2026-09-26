import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import json
import time

from cnn_numpy import SimpleCNN

np.random.seed(42)

# ---- Load public dataset: UCI Optical Recognition of Handwritten Digits ----
digits = load_digits()
X = digits.images / 16.0  # normalize to [0,1]
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_classes = 10
y_train_oh = np.eye(n_classes)[y_train]

model = SimpleCNN(n_filters=8, k=3, dense_hidden=64, n_classes=n_classes, in_size=8)

epochs = 25
batch_size = 32
lr = 0.05
n_train = X_train.shape[0]

history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

y_test_oh = np.eye(n_classes)[y_test]

t0 = time.time()
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(n_train)
    X_train, y_train, y_train_oh = X_train[perm], y_train[perm], y_train_oh[perm]

    epoch_losses = []
    correct = 0
    for i in range(0, n_train, batch_size):
        xb = X_train[i:i + batch_size]
        yb = y_train_oh[i:i + batch_size]
        probs, cache = model.forward(xb)
        loss = -np.mean(np.sum(yb * np.log(probs + 1e-9), axis=1))
        epoch_losses.append(loss)
        correct += np.sum(np.argmax(probs, axis=1) == np.argmax(yb, axis=1))
        model.backward(probs, yb, cache, lr=lr)

    train_loss = float(np.mean(epoch_losses))
    train_acc = float(correct / n_train)

    val_probs, _ = model.forward(X_test)
    val_loss = float(-np.mean(np.sum(y_test_oh * np.log(val_probs + 1e-9), axis=1)))
    val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_test))

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

elapsed = time.time() - t0
print(f"Training time: {elapsed:.1f}s")

# ---- Final evaluation ----
final_probs, _ = model.forward(X_test)
y_pred = np.argmax(final_probs, axis=1)
test_acc = float(np.mean(y_pred == y_test))
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print("\nFinal test accuracy:", test_acc)
print(report)

with open("history.json", "w") as f:
    json.dump({"history": history, "test_accuracy": test_acc,
                "training_time_sec": elapsed}, f, indent=2)

with open("classification_report.txt", "w") as f:
    f.write(f"Final Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

np.save("confusion_matrix.npy", cm)

# ---- Plot 1: Loss curves ----
plt.figure(figsize=(6, 4))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", ms=3)
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", marker="o", ms=3)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()

# ---- Plot 2: Accuracy curves ----
plt.figure(figsize=(6, 4))
plt.plot(history["epoch"], history["train_acc"], label="Train Accuracy", marker="o", ms=3)
plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy", marker="o", ms=3)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.close()

# ---- Plot 3: Confusion matrix ----
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (Test Accuracy = {test_acc*100:.2f}%)")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ---- Plot 4: Sample predictions ----
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
idxs = np.random.choice(len(X_test), 16, replace=False)
for ax, idx in zip(axes.ravel(), idxs):
    ax.imshow(X_test[idx], cmap="gray")
    pred = y_pred[idx]
    true = y_test[idx]
    color = "green" if pred == true else "red"
    ax.set_title(f"P:{pred} / T:{true}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.close()

print("Saved: loss_curve.png, accuracy_curve.png, confusion_matrix.png, sample_predictions.png")
