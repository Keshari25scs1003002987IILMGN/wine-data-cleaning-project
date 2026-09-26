"""
Deep Learning Application in Data Science — Week 5 Task
Handwritten Digit Classification using a Convolutional Neural Network (CNN)

Framework : TensorFlow / Keras
Dataset   : Optical Recognition of Handwritten Digits (UCI / scikit-learn `load_digits`),
            8x8 grayscale images, 10 classes (digits 0-9), 1797 samples.
            Publicly available at: https://archive.ics.uci.edu/dataset/80

Run:
    pip install tensorflow scikit-learn matplotlib
    python cnn_tensorflow.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load and prepare the publicly available dataset
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.images.astype("float32") / 16.0          # normalize pixel range [0, 16] -> [0, 1]
y = digits.target
X = X.reshape(-1, 8, 8, 1)                            # add channel dimension for Conv2D

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ---------------------------------------------------------------------------
# 2. Define the CNN architecture
#    Conv2D -> ReLU -> MaxPool -> Flatten -> Dense -> ReLU -> Dense -> Softmax
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", name="conv2d_1"),
    layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
    layers.Flatten(name="flatten"),
    layers.Dense(64, activation="relu", name="dense_hidden"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax", name="dense_output"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    verbose=2,
)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred)

# ---------------------------------------------------------------------------
# 5. Plots (loss/accuracy curves, confusion matrix)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curve")
plt.tight_layout(); plt.savefig("tf_loss_curve.png", dpi=150)

plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy Curve")
plt.tight_layout(); plt.savefig("tf_accuracy_curve.png", dpi=150)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix (acc={test_acc*100:.2f}%)")
plt.tight_layout(); plt.savefig("tf_confusion_matrix.png", dpi=150)

model.save("digit_cnn_model.keras")
print("Saved model to digit_cnn_model.keras")
