# EMGesture

**A Tkinter desktop app for recording surface-EMG from an Arduino, training scikit-learn classifiers on hand gestures, and running live gesture prediction — all behind a one-click launcher.**

EMGesture gives you the full loop: connect an Arduino that streams analog EMG, capture labeled gesture data through a guided GUI, train a classifier against your own body, then drop into a live testing view that classifies your gestures in real time. No notebooks, no command-line ML, no prior ML setup on your machine — a single `launch.py` creates a virtualenv, installs dependencies, and opens the GUI on first run.

---

## Features

- **One-click launch** on macOS and Windows. First run bootstraps a `.venv`, installs `requirements.txt`, and starts the GUI. Later runs skip straight to the GUI.
- **Dual-channel serial capture** over USB from any Arduino-compatible board that prints analog readings to the serial port at 115200 baud. Channels are named `a0` / `a1`.
- **Label-aware capture** — manual single-gesture, auto-run sequences, or a guided session that walks you through N repeats of every gesture.
- **Per-user calibration** — the "Calibrate Idle" button finds the quietest one-second window in the live buffer and uses it as the noise baseline. Rejects itself if the standard deviation is too high (detects accidental motion during calibration).
- **Auto-best model training** — scikit-learn RandomForest, SVM, MLP, KNN, and LogisticRegression get benchmarked with `GroupShuffleSplit` (no within-capture leakage), the winner by macro-F1 is saved. Manual pick is also supported.
- **26 time-domain features** including MAV, RMS, WL, ZC, SSC, Willison amplitude, IAV, log-detector, difference MAV/std, AR(4) coefficients, peak, peak-to-peak, duration, and **5 tokenized burst-shape features** that split the event slice into 5 sub-windows and encode where the peak sits and how concentrated the activity is. The tokenized features are specifically designed to separate transient gestures (snap finger) from sustained ones (fist, open).
- **Live event detection** — per-poll, the predictor centers the current 3-second window against a robust 10th-percentile baseline, computes a smoothed envelope, applies a `3σ` activity threshold derived from calibration noise, and finds the latest active run in the mask. Classification fires on every new or growing slice; majority-vote smoothing across a 5-poll window reduces flicker.
- **Debug Mode** — one checkbox auto-captures full predictor state dumps at 2 Hz into a 200-entry ring buffer. One more click exports everything (training classification report, bundle config, calibration, live buffer stats, fresh + historical debug dumps, prediction history, log tail) to a single `.txt` file you can share for debugging.
- **User Data Browser** with raw / processed / overlay views, feature summary per capture, and a **Compare** mode that splits the plot into a 2-row view so you can overlay two captures side by side (even from different users).
- **Cross-platform entry points** — `setup_mac.command` (Finder double-click) and `setup_windows.bat` (Explorer double-click) both shim to `launch.py`, which also runs directly from VS Code's "Run Python File".

---

## Hardware

- **Microcontroller**: Arduino Uno / Nano / ESP32 / any board that can print to USB serial at 115200 baud
- **EMG sensor**: 1 or 2 surface EMG modules (MyoWare 2.0, Grove-EMG, or equivalent) wired to `A0` and optionally `A1`
- **Target application**: hand and finger gestures on the forearm — `fist`, `open`, `one`, `thumb`, `snap_finger`, plus custom labels you add in the GUI

The Arduino sketch should print lines to serial that the GUI can parse (one or two channels supported). `adc` is converted to volts using a 5.0 V reference and 1023-step ADC.

### Arduino serial protocol

The GUI listens at **115200 baud** and accepts one sample per line. Supported
formats:

```text
adc0
adc0 adc1
t_ms,adc0,voltage0
t_ms,adc0,voltage0,adc1,voltage1
```

- `adc0` / `adc1` are raw 10-bit ADC readings from `A0` / `A1`.
- `t_ms` is device time in milliseconds; the GUI converts it to seconds.
- If the Arduino sends raw ADC only, the GUI timestamps the sample on the host
  computer and computes voltage as `adc / 1023 * 5.0`.
- If the Arduino sends timed comma-separated samples, the voltage fields are
  used directly.

There is currently no Arduino `.ino` sketch committed in this repository. If a
sketch is added later, place it under an `arduino/` directory and keep its serial
output aligned with the formats above.

---

## Quick start

### macOS (Finder double-click)

1. Clone the repo or download as ZIP
2. Open the folder in Finder
3. Double-click `setup_mac.command`
4. Wait for the first-run bootstrap (creates `.venv`, installs `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyserial`, `joblib`)
5. The GUI window opens

### Windows (Explorer double-click)

1. Clone the repo
2. Double-click `setup_windows.bat`
3. First run installs dependencies into a local `.venv`
4. The GUI opens

### VS Code (Run Python File)

1. Open the project folder in VS Code
2. Open `launch.py`
3. Click the "Run Python File" button (top-right play icon)
4. Same bootstrap, same GUI

### Terminal

```bash
cd /path/to/emg
python3 launch.py
```

**Force a clean reinstall**: delete the `.venv` folder and run any of the above again.

---

## Typical workflow

### 1. Connect your Arduino

Plug in the board, wait for it to appear on a serial port (e.g., `/dev/cu.usbserial-*` on macOS or `COM3` on Windows).

### 2. Calibrate

1. Go to **Training → Single Gesture Collection** (or **Testing**)
2. Enter your user name (e.g., `eric`)
3. Select the serial port, click **Connect**
4. With your hand **completely at rest on the table**, click **Calibrate Idle**
5. The calibration status line should show a mean voltage in `0.05–0.30 V` and a standard deviation **below 0.03 V**. If it's above, the Calibrate Idle call will refuse to save — just click it again with your hand still.

### 3. Record gestures

1. Pick a label (e.g., `fist`)
2. Click **Start Capture** or **Auto Run** (records N captures back-to-back)
3. Do the gesture when the banner says `DO FIST NOW`, relax when it says `REST`
4. Repeat for every label you want
5. Suggested target: **30–50 captures per label** for a stable model

### 4. Train

1. Go to **Home → Train Model**
2. Pick the user, labels, and channels to include
3. Set **Model Type** to **Auto Best** (or manually pick RF / SVM / MLP / KNN / LogReg)
4. Give the model a name and click **Train**
5. Read the classification report on the right — per-class precision, recall, F1. Realistic expectations for single-channel surface EMG with ~30 captures per class: **macro F1 around 0.7–0.9**. F1 ≈ 1.00 is usually a red flag for leakage or overfitting.

### 5. Test live

1. Go to **Home → Test Model**
2. Connect to the serial port
3. Pick the trained model, click **Run Test**
4. Do a gesture — the big **Predicted** label should flip between `idle` and the gesture name
5. If something looks off, toggle **Debug Mode** on, repeat your test, then click **Export Debug Report** and open the saved `.txt`

---

## Architecture

```
emg/
├── launch.py               # bootstrap launcher (stdlib only; creates .venv then relaunches)
├── setup_mac.command       # Finder double-click shim → calls launch.py
├── setup_windows.bat       # Explorer double-click shim → calls launch.py
├── requirements.txt        # numpy, pandas, scikit-learn, matplotlib, pyserial, joblib
│
├── emg_data_gui.py         # Tkinter GUI, serial reader, live plotting, all user workflows
├── emg_model_tools.py      # feature extraction, event detection, training, LivePredictor
├── emg_data_tools.py       # CSV I/O, label catalog, calibration save/load, capture records
│
└── database/               # created by the app at runtime
    ├── label_catalog.json  # tracked — shared label definitions
    ├── users/              # per-user captures (gitignored by default)
    ├── calibrations/       # per-user noise floors (gitignored)
    ├── models/             # trained .joblib bundles (gitignored)
    └── processed_previews/ # export cache (gitignored)
```

### Signal pipeline

1. `poll_serial` reads the USB port every 20 ms, parses each line, appends `(timestamp, adc, voltage)` tuples to per-channel `deque`s. Buffers are trimmed to the last 5 seconds for plotting and 3 seconds for prediction.
2. `LivePredictor._predict_single` collects the latest buffer window, estimates sampling rate from `time_s` in **float64** (earlier versions used float32, which silently rounded Unix timestamps to the same value), runs `analyze_model_window`, and either returns `idle` / `capturing` or fires the classifier.
3. `center_signal` subtracts a **per-signal 10th-percentile baseline** (robust to electrode drift between recording sessions and live use) — this is the critical fix that keeps training-time and inference-time feature distributions aligned.
4. `_smooth_envelope` low-passes `|centered|` with a short moving average, `_event_mask_for_signal` thresholds it at `max(3 × noise_std, 0.02 V)`.
5. `_latest_active_slice` finds the most recent contiguous run of active samples that is at least 80 ms long.
6. `extract_feature_vector` computes 26 features on the slice; per-channel features are concatenated, scaled by the bundled `StandardScaler`, and classified.
7. `_commit_prediction` applies majority voting and an `unknown` vs `capturing` display-threshold policy (first few low-confidence polls of a new gesture display as `capturing`, not `unknown`, to avoid flicker).

---

## Debug Mode

When gestures don't classify correctly, turn on **Debug Mode** in the Testing view. Every 500 ms the GUI auto-captures a full internal state snapshot into a ring buffer:

- estimated sampling rate
- baseline used (from calibration or 10th-percentile fallback)
- event detection threshold
- per-channel raw / centered / envelope stats
- mask run lengths (to spot fragmentation)
- event slice boundaries and lag
- classifier prediction + confidence
- which `display_prediction` would be shown

Click **Export Debug Report** to dump all snapshots, plus the loaded model bundle's classification report, calibration state, bundle config, recent predictions, and the log tail into a single `.txt` file. One file that a collaborator can open and immediately see what the predictor is seeing versus what it should be seeing.

---

## Troubleshooting

**Predictor is stuck on `capturing` forever**
The event-end detection is too sensitive. Enable Debug Mode, export a report, and check the `tail_active_fraction` and `mask_last_sample_ON` values — if the tail is always active, your calibration's `voltage_std` is probably inflated (Calibrate Idle was clicked while you were moving). Redo Calibrate Idle with your hand completely still.

**Training F1 is ≈1.00 but live accuracy is terrible**
This was a real bug: when the training calibration's `voltage_mean` didn't match the captured files' rest levels (because those files were recorded at a different session), the training pipeline kept the entire capture as "active" while the live pipeline correctly sliced to the gesture only — two completely different feature distributions. Fixed by switching `_baseline_mean` to always use a per-signal 10th-percentile regardless of calibration.

**Only `fist` classifies well, other gestures are random**
Surface EMG with one channel on the forearm has a hard time separating `open`, `one`, and `thumb` because they all use weaker extensors / individual finger muscles with overlapping projections. Either (a) add a second channel on a different muscle group, (b) collect more captures per class (50–100 each), or (c) start with a smaller label set (`fist` vs `open` vs `rest`) and grow it once that works.

**`unknown` flashes before the real prediction**
The very first poll of a new gesture only sees ~100 ms of signal and is understandably low-confidence. EMGesture now shows `capturing` during the first 1–2 low-confidence polls, and only commits to `unknown` after 3 consecutive sub-threshold polls (≈600 ms). If you still see `unknown` flashes, lower the **Confidence threshold** slider in the Testing view.

**The GUI never launches / launch.py errors on import**
Delete the `.venv` folder and run `launch.py` / `setup_mac.command` again to rebuild the virtualenv from scratch.

---

## Requirements

From `requirements.txt`:

```
joblib>=1.3
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
pyserial>=3.5
```

Tkinter ships with the Python from python.org on macOS/Windows. On Homebrew Python you may need `brew install python-tk`.

---

## Status

Single-mode gesture recognition only. The older "continuous" streaming mode has been removed. The project is under active experimentation and the signal-processing pipeline has several moving pieces (calibration, baseline estimation, event slicing, tokenized features) that sometimes need to be retuned together when switching users or hardware.
