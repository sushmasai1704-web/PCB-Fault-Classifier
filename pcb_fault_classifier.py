"""
PCB Fault Classifier — ML-Based Automated Fault Detection
Author: Mellamputi Sai Sushma
Description:
    Extracts frequency-domain and statistical features from PCB power rail signals
    and trains a Random Forest classifier to automatically detect fault types.
    Dataset includes realistic signal overlap between classes to reflect
    real-world measurement conditions.

Fault Classes:
    0 — Normal          : Clean power rail
    1 — Mains Hum       : 50 Hz power line interference
    2 — Switching Noise : DC-DC converter harmonics
    3 — Ground Bounce   : Broadband noise + random spikes
    4 — DC Ripple       : Periodic ripple on DC rail
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

FAULT_LABELS = {
    0: 'Normal',
    1: 'Mains Hum',
    2: 'Switching Noise',
    3: 'Ground Bounce',
    4: 'DC Ripple',
}

FS  = 10000
DUR = 0.5
N   = int(FS * DUR)


# ─────────────────────────────────────────────
#  Realistic Signal Simulator
#  Key: overlapping SNR ranges across classes
#  so the classifier sees genuinely hard cases
# ─────────────────────────────────────────────

def simulate_signal(fault_class, rng):
    t    = np.linspace(0, DUR, N, endpoint=False)
    # High noise floor — makes classes harder to separate
    nlvl = rng.uniform(0.08, 0.30)
    base = nlvl * rng.standard_normal(N)

    if fault_class == 0:  # Normal
        sig = 3.3 + base * 0.6
        # Inject faint 50Hz pickup 30% of the time (real boards do this)
        if rng.random() < 0.30:
            sig += rng.uniform(0.02, 0.06) * np.sin(2*np.pi*50*t)

    elif fault_class == 1:  # Mains Hum
        # Amplitude range deliberately overlaps with "normal + faint hum"
        amp = rng.uniform(0.05, 0.35)
        ph  = rng.uniform(0, 2*np.pi)
        sig = 3.3 + amp * np.sin(2*np.pi*50*t + ph)
        sig += rng.uniform(0.1, 0.4) * amp * np.sin(2*np.pi*150*t)
        sig += base

    elif fault_class == 2:  # Switching Noise
        # Frequency range overlaps with DC Ripple at lower end
        f_sw = rng.uniform(120, 500)
        amp  = rng.uniform(0.05, 0.20)
        sig  = 3.3 + amp * np.sin(2*np.pi*f_sw*t)
        for h in range(2, 5):
            sig += (amp/h/1.5) * np.sin(2*np.pi*f_sw*h*t + rng.uniform(0, np.pi))
        sig += base

    elif fault_class == 3:  # Ground Bounce
        sig = 3.3 + rng.uniform(0.08, 0.22) * rng.standard_normal(N)
        # Variable spike count — sparse spikes look like normal
        n_spikes = rng.integers(2, 25)
        idx = rng.integers(0, N, n_spikes)
        sig[idx] += rng.choice([-1,1], n_spikes) * rng.uniform(0.10, 0.60, n_spikes)
        if rng.random() < 0.25:
            sig += rng.uniform(0.02,0.08) * np.sin(2*np.pi*rng.uniform(40,80)*t)

    elif fault_class == 4:  # DC Ripple
        # Wide freq range: 40-200Hz overlaps with Mains Hum at low end
        f_rip = rng.uniform(40, 200)
        amp   = rng.uniform(0.05, 0.28)
        sig   = 3.3 + amp * np.sin(2*np.pi*f_rip*t + rng.uniform(0, 2*np.pi))
        sig  += base

    return sig


# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def extract_features(signal, fs=FS):
    ac = signal - np.mean(signal)
    Nf = len(ac)

    f = {}
    f['rms']       = np.sqrt(np.mean(ac**2))
    f['std']       = np.std(ac)
    f['peak']      = np.max(np.abs(ac))
    f['crest']     = f['peak'] / (f['rms'] + 1e-9)
    f['skewness']  = float(skew(ac))
    f['kurtosis']  = float(kurtosis(ac))

    win = np.hanning(Nf)
    fv  = np.fft.rfft(ac * win)
    fr  = np.fft.rfftfreq(Nf, 1/fs)
    mag = np.abs(fv) * 2 / Nf

    dom         = np.argmax(mag[1:]) + 1
    f['dom_freq']      = fr[dom]
    f['dom_mag']       = mag[dom]
    f['spec_centroid'] = np.sum(fr * mag) / (np.sum(mag) + 1e-9)
    pn = mag / (np.sum(mag) + 1e-9)
    f['spec_entropy']  = -np.sum(pn * np.log(pn + 1e-9))

    f0 = f['dom_freq']
    if f0 > 5:
        v1 = mag[dom]
        hp = sum(mag[int(h*f0*Nf/fs)]**2
                 for h in range(2,5) if int(h*f0*Nf/fs) < len(mag))
        f['thd'] = np.sqrt(hp) / (v1 + 1e-9)
    else:
        f['thd'] = 0.0

    fp, psd = welch(ac, fs=fs, nperseg=min(256, Nf//2))
    tot = np.trapezoid(psd, fp) + 1e-12
    def bp(lo, hi):
        m = (fp >= lo) & (fp < hi)
        return np.trapezoid(psd[m], fp[m]) / tot

    f['pwr_0_80']    = bp(0,   80)
    f['pwr_80_200']  = bp(80,  200)
    f['pwr_200_500'] = bp(200, 500)
    f['pwr_500plus'] = bp(500, fs/2)
    f['spike_ratio'] = f['peak'] / (np.percentile(np.abs(ac), 99) + 1e-9)

    return list(f.values())


FEATURE_NAMES = [
    'RMS', 'Std Dev', 'Peak', 'Crest Factor', 'Skewness', 'Kurtosis',
    'Dom Freq', 'Dom Magnitude', 'Spectral Centroid', 'Spectral Entropy',
    'THD', 'Power 0-80Hz', 'Power 80-200Hz', 'Power 200-500Hz',
    'Power 500Hz+', 'Spike Ratio'
]


def build_dataset(n_per_class=350, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    print(f"Building dataset: {n_per_class} samples × {len(FAULT_LABELS)} classes...")
    for cls in FAULT_LABELS:
        for _ in range(n_per_class):
            X.append(extract_features(simulate_signal(cls, rng)))
            y.append(cls)
    X, y = np.array(X), np.array(y)
    print(f"Dataset: {X.shape}  labels: {np.bincount(y)}")
    return X, y


def train_and_evaluate(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_split=8, min_samples_leaf=3,
            max_features='sqrt', random_state=42, n_jobs=-1))
    ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring='accuracy')
    print(f"\n5-Fold CV : {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = clf.score(X_te, y_te)
    print(f"Test Acc  : {acc*100:.1f}%")
    print("\n" + classification_report(
        y_te, y_pred, target_names=list(FAULT_LABELS.values())))
    return clf, X_te, y_te, y_pred, scores


def plot_results(clf, X_te, y_te, y_pred, cv_scores):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        'PCB Fault Classifier — Results\n'
        'Random Forest (100 trees, depth=8)  |  16 features  |  5-class  |  fs=10kHz',
        fontsize=11, y=1.01)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)
    colors = ['#1f77b4','#2ca02c','#d62728','#ff7f0e','#9467bd']
    rng = np.random.default_rng(7)
    t   = np.linspace(0, DUR*1000, N)

    # Sample waveforms
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (cls, col) in enumerate(zip(FAULT_LABELS, colors)):
        sig = simulate_signal(cls, rng)
        ac  = sig - np.mean(sig)
        ax1.plot(t[:700], ac[:700] + i*1.4, color=col, lw=0.8,
                 label=FAULT_LABELS[cls])
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (V, offset)')
    ax1.set_title('Sample Signals per Fault Class')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # CV scores
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar([f'Fold {i+1}' for i in range(5)],
                   cv_scores*100, color='steelblue', edgecolor='white', lw=0.5)
    ax2.axhline(cv_scores.mean()*100, color='red', lw=1.2, ls='--',
                label=f'Mean={cv_scores.mean()*100:.1f}%')
    ax2.set_ylim(75, 100); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('5-Fold Cross-Validation')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, cv_scores):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                 f'{v*100:.1f}%', ha='center', fontsize=8)

    # Confusion matrix
    ax3 = fig.add_subplot(gs[1, :2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_te, y_pred),
        display_labels=list(FAULT_LABELS.values()))
    disp.plot(ax=ax3, colorbar=False, cmap='Blues')
    ax3.set_title('Confusion Matrix (Test Set)')
    ax3.tick_params(axis='x', rotation=18)

    # Feature importance
    ax4 = fig.add_subplot(gs[1, 2])
    imp = clf.named_steps['rf'].feature_importances_
    idx = np.argsort(imp)[-10:]
    ax4.barh(range(10), imp[idx], color='#2ca02c', edgecolor='white', lw=0.4)
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([FEATURE_NAMES[i] for i in idx], fontsize=8)
    ax4.set_xlabel('Importance'); ax4.set_title('Top 10 Feature Importances')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('sample_output.png', dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Plot saved → sample_output.png")


def classify_signal(signal, clf):
    feat  = np.array(extract_features(signal)).reshape(1, -1)
    pred  = clf.predict(feat)[0]
    proba = clf.predict_proba(feat)[0]
    return FAULT_LABELS[pred], round(proba[pred]*100, 1)


if __name__ == "__main__":
    print("=" * 50)
    print("  PCB Fault Classifier")
    print("=" * 50)
    X, y = build_dataset(n_per_class=350)
    clf, X_te, y_te, y_pred, cv_scores = train_and_evaluate(X, y)
    plot_results(clf, X_te, y_te, y_pred, cv_scores)

    print("\n── Inference demo ──")
    rng = np.random.default_rng(99)
    for cls in FAULT_LABELS:
        sig         = simulate_signal(cls, rng)
        label, conf = classify_signal(sig, clf)
        true        = FAULT_LABELS[cls]
        mark        = "✅" if label == true else "❌"
        print(f"  {mark}  True: {true:<18} Predicted: {label:<18} Conf: {conf}%")
