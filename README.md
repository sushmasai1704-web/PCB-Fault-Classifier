# PCB Fault Classifier

> ML-based automated fault detection for PCB power rails — classifies 5 fault types from signal features using Random Forest.  
> Eliminates manual oscilloscope inspection by automating frequency-domain fault diagnosis.

---

## Problem It Solves

Hardware validation engineers spend significant time manually inspecting PCB power rail signals on oscilloscopes to identify fault types — mains hum, switching noise, ground bounce, DC ripple. This classifier automates that process: given a raw voltage signal, it extracts 16 signal features and predicts the fault class with a confidence score in milliseconds.

Motivated by real PCB validation work at Smile Electronics, where manual DFT-based fault detection was applied across 25+ boards. This project automates that workflow using ML — reducing time-to-diagnosis and enabling batch screening.

**Directly relevant to:** Hardware Validation · PCB Bring-up · Test Automation · Signal Integrity · IoT Diagnostics

---

## Fault Classes

| Class | Description | Key Signal Characteristic |
|-------|-------------|--------------------------|
| Normal | Clean power rail | Low RMS, no dominant frequency |
| Mains Hum | 50 Hz power line coupling | Strong 50/150 Hz harmonics |
| Switching Noise | DC-DC converter harmonics | High-frequency periodic content |
| Ground Bounce | Broadband noise + spikes | High kurtosis, high spike ratio |
| DC Ripple | Periodic ripple on DC rail | Single dominant 80–180 Hz tone |

---

## ML Pipeline

```
Raw PCB Signal
 → Feature Extraction (16 features: statistical + spectral + PSD-based)
 → StandardScaler normalization
 → Random Forest Classifier (150 trees, depth=12)
 → Fault Class + Confidence Score
```

**Features extracted:**
- Statistical: RMS, std, peak, crest factor, skewness, kurtosis
- Spectral: dominant frequency, spectral centroid, spectral entropy, THD estimate
- PSD bands: power fractions in 0–80 Hz, 80–200 Hz, 200–500 Hz, 500 Hz+
- Spike ratio: peak-to-99th-percentile ratio

---

## Results

| Metric | Score |
|--------|-------|
| 5-Fold CV Accuracy | 100.00% ± 0.00% |
| Test Set Accuracy | 100.00% |
| Precision (all classes) | 1.00 |
| Recall (all classes) | 1.00 |

### Why 100% — and what it means

The perfect score is a direct consequence of the dataset design, not a claim about production generalization.

Each fault class was generated with a distinct, physically motivated spectral signature: mains hum produces strong energy exclusively at 50/150 Hz, switching noise produces high-frequency periodic content, ground bounce produces broadband kurtosis with spike bursts. These signatures are **spectrally non-overlapping by construction** — the 16 extracted features separate them cleanly in feature space, and Random Forest exploits this completely.

In other words: the classifier works perfectly *because the problem is well-posed*, not because the model is over-engineered. A linear SVM achieves the same result on this dataset.

**What this project demonstrates:**
- Signal processing pipeline design (FFT, PSD, statistical feature extraction)
- Feature engineering that encodes domain knowledge about fault physics
- End-to-end ML workflow from raw waveform to classified output
- The architecture is sound for real data — real oscilloscope captures would introduce noise overlap between classes, requiring more training data and likely a softer decision boundary

**What it does not claim:**
- Generalization to unseen hardware with different noise floors, cable lengths, or PCB layouts
- Production-ready accuracy on real captured signals without retraining

---

## Run It

```bash
git clone https://github.com/sushmasai1704-web/PCB-Fault-Classifier
cd PCB-Fault-Classifier
pip install -r requirements.txt
python pcb_fault_classifier.py
```

**Classify your own signal:**

```python
from pcb_fault_classifier import classify_signal, build_dataset, train_and_evaluate
import numpy as np

# Train model
X, y = build_dataset()
clf, *_ = train_and_evaluate(X, y)

# Classify a new signal (your oscilloscope data as numpy array)
signal = np.loadtxt("my_pcb_capture.csv", delimiter=",")[:, 1]
label, confidence = classify_signal(signal, clf)
print(f"Fault: {label} | Confidence: {confidence:.1f}%")
```

---

## Tech Stack

`Python 3.10` `NumPy` `SciPy` `scikit-learn` `Matplotlib`

## Skills Demonstrated

`Signal Processing` `Feature Engineering` `FFT/DFT` `PSD Analysis` `Random Forest` `scikit-learn` `Hardware Validation` `Python`

---

## Author

**M Sai Sushma**  
[github.com/sushmasai1704-web](https://github.com/sushmasai1704-web)
