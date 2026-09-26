# Deep Learning Application in Data Science — Week 5 Task

Handwritten digit classification using a Convolutional Neural Network (CNN),
built as part of the *Virtual Data Science with Python* internship (YuvaIntern).

## Problem Statement
Classify 8x8 grayscale images of handwritten digits (0–9) into their correct
class using a CNN — a multi-class **classification** problem.

## Dataset
- **Name:** Optical Recognition of Handwritten Digits Dataset (UCI ML Repository)
- **Access:** Public, bundled with scikit-learn (`sklearn.datasets.load_digits`)
  and mirrored at https://archive.ics.uci.edu/dataset/80
- **Size:** 1,797 samples, 8x8 pixels, pixel values 0–16, 10 balanced classes

## Architecture
```
Input (8x8x1)
  -> Conv2D(8 filters, 3x3) + ReLU
  -> MaxPooling2D(2x2)
  -> Flatten
  -> Dense(64) + ReLU
  -> Dropout(0.3)
  -> Dense(10) + Softmax
```
See `architecture_diagram.png`.

## Files
| File | Description |
|---|---|
| `cnn_tensorflow.py` | Primary deliverable — full Keras/TensorFlow implementation (run this) |
| `cnn_numpy.py` / `train.py` | From-scratch NumPy reference implementation used to validate the architecture and generate the results in this repo/report |
| `history.json` | Per-epoch training/validation loss & accuracy |
| `classification_report.txt` | Precision/recall/F1 per class |
| `*.png` | Architecture diagram, loss/accuracy curves, confusion matrix, sample predictions |
| `Week5_DeepLearning_Report.docx` | Full project report |

## How to run (TensorFlow version)
```bash
pip install tensorflow scikit-learn matplotlib
python cnn_tensorflow.py
```

## Results (NumPy reference run, 25 epochs)
- **Test Accuracy:** 95.83%
- **Macro F1-score:** 0.958
- Training time: ~11 seconds on CPU

## Key Challenges
- Implementing convolution and max-pooling backward passes manually (no
  autograd) required careful gradient bookkeeping.
- With only ~1,800 samples, the model can overfit quickly — mitigated with
  a modest architecture, a hidden layer of 64 units, and (in the Keras
  version) dropout.
- Small 8x8 inputs limit receptive field growth, so only a single conv/pool
  stage was used before flattening.

## Author
Keshari — YuvaIntern Virtual Data Science with Python Trainee
