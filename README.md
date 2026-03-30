# Sparse EMG Signal Processing Pipeline
### For Low-Power Microcontroller-Based Finger Gesture Classification and Tele-Operated Robotics

---

## Overview

This pipeline processes EMG signals from a wrist-worn data glove to classify individual finger movements and predict future gesture commands for tele-operated robots. The core design principle is **compressed sensing** — all heavy computation (optimization, decomposition, training) happens offline on a PC. The microcontroller runs only lightweight inference using precomputed matrices and coefficients.

**Target hardware:** Arduino UNO R4, STM32, ESP32 (any ARM Cortex-M class MCU)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (PC / Workstation)                   │
│                                                                   │
│  [1] EMG Acquisition ──► [2] Wavelet Transform                   │
│         x(t) = Σhᵢ*sᵢ+n      W = Ψ·x,  ||W||₀ << N             │
│                  │                   │                            │
│                  ▼                   ▼                            │
│         [3] Sparse Optimization (Bregman + ADMM)                 │
│              min ||W||₁  s.t.  AW = y                            │
│              Wᵏ⁺¹ = shrink(W̃ + bᵏ, 1/μ)                        │
│              uᵏ⁺¹ = uᵏ + AWᵏ⁺¹ - zᵏ⁺¹                          │
│                  │                                                │
│                  ▼                                                │
│         [4] Source Separation (ICA / PCA)                        │
│              s = W·x  (unmixing matrix W precomputed)            │
│                  │                                                │
│                  ▼                                                │
│         [5] Finger Classification by Frequency Band              │
│              fingerₖ = argmax_i { ∫_Bk Sᵢ(f) df }               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                           │
              [ Store: W_unmix, Φ_sparse, aₙ, bₙ coefficients ]
                           │
┌─────────────────────────────────────────────────────────────────┐
│                  ONLINE (Microcontroller)                         │
│                                                                   │
│         [6] Fourier Series Prediction                            │
│              x̂(t) = a₀/2 + Σₙ [aₙcos(nωt) + bₙsin(nωt)]       │
│                  │                                                │
│                  ▼                                                │
│         [7] Hypothesis Testing                                   │
│              H₀: signal ~ Noise    H₁: signal ~ Gesture          │
│              T = (x̄ - μ₀) / (s/√n)                              │
│              Reject H₀ if |T| > t_{α/2, n-1}                    │
│                  │                     │                          │
│            H₁ accepted           H₀ not rejected                 │
│                  │                     │                          │
│                  ▼                     └──► back to Stage 3      │
│         Transmit to Robot                   (iterate, max 10x)   │
│         (Fourier coefficients only)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Explanation

### Stage 1 — EMG Acquisition
- Data glove with surface EMG electrodes placed on the wrist
- Captures superposition of motor unit action potentials (MUAPs)
- Signal model: `x(t) = Σᵢ hᵢ(t) * sᵢ(t) + n(t)`
  - `hᵢ` = MUAP waveform of i-th motor unit
  - `sᵢ` = firing train
  - `n(t)` = sensor noise

### Stage 2 — Wavelet Transform (Sparsification)
- Apply Discrete Wavelet Transform (DWT) using Daubechies db4/db6 basis
- EMG signals are naturally sparse in the wavelet domain
- Formula: `W = Ψ · x`
- Sparsity condition: `||W||₀ << N` (most coefficients near zero)
- Complexity: **O(N)** — cheaper than FFT's O(N log N)
- Achievable compression: **16x+** with SNR > 60 dB

### Stage 3 — Sparse Optimization (Bregman + ADMM)
Solves the L1-minimization (basis pursuit) problem using two complementary methods:

**Basis Pursuit:**
```
min ||W||₁  subject to  AW = y
where A = ΦΨ⁻¹  (sensing matrix, M×N, M << N)
```

**Bregman Splitting:**
```
Wᵏ⁺¹ = shrink(W̃ + bᵏ,  1/μ)
bᵏ⁺¹ = bᵏ + (y - AWᵏ⁺¹)
shrink(x, t) = sign(x) · max(|x| - t, 0)
```

**Augmented Lagrangian (ADMM):**
```
W-update:  Wᵏ⁺¹ = (AᵀA + ρI)⁻¹ Aᵀ(y - uᵏ)   ← precomputed offline
z-update:  zᵏ⁺¹ = shrink(AWᵏ⁺¹ + uᵏ, 1/ρ)
u-update:  uᵏ⁺¹ = uᵏ + AWᵏ⁺¹ - zᵏ⁺¹
```

> **Key insight:** The matrix inverse `(AᵀA + ρI)⁻¹` is computed **once offline** and stored. At runtime it is a single matrix-vector multiply.

### Stage 4 — Source Separation (ICA / PCA)
- **ICA preferred** because each finger's motor commands are physiologically independent sources
- ICA model: `x = A_mix · s`,  recover `s = W_unmix · x`
- Independence criterion: `I(s₁, s₂, s₃, s₄, s₅) → 0`
- PCA fallback: `X = UΣVᵀ`,  project: `z = Vᵀx`
- Unmixing matrix `W_unmix` computed offline, stored in Flash
- Runtime: **O(5N)** — one matrix-vector multiply

### Stage 5 — Finger Classification by Frequency Band
- Analyse power spectral density of each separated source
- Assign each source to a finger based on dominant frequency band
```
Sᵢ(f) = |Fᵢ(f)|²
fingerₖ = argmax_i { ∫_Bk Sᵢ(f) df }
f_mean  = ∫ f·S(f) df / ∫ S(f) df
```
- EMG frequency range: 20–500 Hz with finger-specific spectral peaks

### Stage 6 — Fourier Series Prediction
- Predicts the next signal window using stored Fourier coefficients
- Trained offline on historical gesture windows
```
x̂(t) = a₀/2 + Σₙ [aₙcos(nωt) + bₙsin(nωt)]

aₙ = (2/T) ∫₀ᵀ x(t) cos(nωt) dt   ← computed offline
bₙ = (2/T) ∫₀ᵀ x(t) sin(nωt) dt   ← stored in Flash
```
- Runtime prediction: evaluate trig sum over K terms — **O(5K)** operations
- **Bandwidth reduction:** transmit K coefficients instead of N samples
  - For N=256, K=20: **12.8x** reduction in transmitted data

### Stage 7 — Hypothesis Testing (Quality Gate)
```
H₀: x̂(t) ~ Noise(μ₀, σ₀²)       signal is not a valid gesture
H₁: x̂(t) ~ Signal(μ₁, σ₁²)      signal encodes a confident gesture

Test statistic:  T = (x̄ - μ₀) / (s / √n)
Decision rule:   Reject H₀  if  |T| > t_{α/2, n-1}   (α = 0.05)
Alternative:     χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ
```

| Result | Action |
|--------|--------|
| H₁ accepted | Transmit Fourier coefficients to robot |
| H₀ not rejected | Return to Stage 3, iterate (max 10 iterations) |
| Max iterations reached | Send null/hold command to robot |

---

## Why This Is Better Than Current Methods

| Method | MCU Deployable | Predicts Future | Compresses TX | Samples Needed |
|--------|---------------|-----------------|---------------|----------------|
| CNN / CNN-LSTM | ❌ GPU required | ❌ | ❌ | Large (10k+) |
| Transformer (CT-HGR) | ❌ GPU required | ❌ | ❌ | Large |
| Multi-stream TCN+LSTM | ❌ Raspberry Pi+ | ❌ | ❌ | Very large |
| SVM / k-NN | ⚠️ Marginal | ❌ | ❌ | Moderate |
| Prior CS + OMP | ✅ 16-bit MCU | ❌ | ✅ | Low |
| **This pipeline** | ✅ Arduino/STM32 | ✅ Fourier | ✅ 12x | **Very low** |

### Key advantages over deep learning:
1. **No GPU at inference time** — training is offline; runtime is matrix multiplies and trig sums
2. **Fewer training samples** — compressed sensing exploits sparsity, not data volume
3. **Predictive** — Fourier extrapolation anticipates the next gesture (no existing EMG-robot pipeline does this)
4. **Bandwidth efficient** — transmits coefficients, not raw waveforms
5. **Self-correcting** — hypothesis gate ensures only confident signals reach the robot
6. **Interpretable** — ICA components map to physiological finger sources, not black-box features

---

## Microcontroller Runtime Analysis

All operations on the MCU for N=256, K=20, 5 fingers:

| Stage | Operation | Approx. Ops |
|-------|-----------|-------------|
| Sparse projection | Φ·x (M=64 rows) | ~1,280 |
| ICA unmixing | W·x | ~1,280 |
| Frequency binning | argmax over bands | ~5 |
| Fourier prediction | Σ aₙcos + bₙsin | ~100 |
| Hypothesis test | T-statistic + lookup | ~20 |
| **Total** | | **~2,685 ops** |

At 48 MHz (Arduino UNO R4 ARM Cortex-M4 with FPU):
- ~2,685 ops ≈ **56 microseconds**
- EMG window size: 100–200 ms
- **CPU utilization: < 0.1%**

### Memory footprint:

| Item | Size |
|------|------|
| Wavelet projection matrix Φ (sparse) | ~6 KB |
| ICA unmixing matrix W (5×256) | ~5 KB |
| Fourier coefficients (aₙ, bₙ, 5 fingers, K=20) | ~800 B |
| Hypothesis test t-table | ~200 B |
| Signal buffer (runtime) | ~1 KB |
| **Total** | **< 15 KB Flash, < 2 KB SRAM** |

Arduino UNO R4 has 256 KB Flash and 32 KB SRAM — **comfortably within limits**.

---

## Novelty Summary

| # | Contribution |
|---|---|
| N1 | First end-to-end sparse EMG pipeline deployable on bare microcontrollers |
| N2 | Bregman splitting + ADMM applied to EMG (prior work uses OMP/BSBL) |
| N3 | ICA-based per-finger decomposition grounded in physiological independence |
| N4 | Fourier series prediction for gesture anticipation in tele-operation |
| N5 | Hypothesis-gated feedback loop as a statistical quality control mechanism |
| N6 | 12x+ bandwidth reduction in the robot command channel |

---

## Proposed Paper Title (if submitting)

> *"A Compressed Sensing Framework with Bregman-ADMM Optimization and Fourier Prediction for EMG-Based Finger Gesture Teleoperation on Resource-Constrained Microcontrollers"*

---

## Folder Structure (suggested implementation)

```
emg-sparse-pipeline/
├── offline/
│   ├── 01_acquire.py          # Data glove EMG capture
│   ├── 02_wavelet.py          # DWT sparsification
│   ├── 03_admm_optimize.py    # Bregman + ADMM solver
│   ├── 04_ica_decompose.py    # FastICA / PCA source separation
│   ├── 05_classify_fingers.py # Frequency-band finger assignment
│   ├── 06_fourier_fit.py      # Fit Fourier coefficients per finger
│   └── export_weights.py      # Export matrices as C headers
├── firmware/                  # Arduino / STM32 code
│   ├── inference.ino
│   ├── weights.h              # Precomputed matrices (auto-generated)
│   └── hypothesis_test.h
├── robot/
│   └── receive_and_actuate.py # Robot-side command decoder
└── README.md
```

---

## References / Related Work

- Split Bregman method: Goldstein & Osher (2009), SIAM J. Imaging Sciences
- ADMM: Boyd et al. (2011), Foundations and Trends in Machine Learning
- EMG compressed sensing: Dixon et al., IEEE Trans. Biomedical Circuits and Systems
- FastICA: Hyvärinen & Oja (2000), Neural Networks
- CT-HGR Transformer: recent NeurIPS/ICASSP proceedings (2023–2024)
- NinaPro EMG benchmark dataset: Atzori et al.
