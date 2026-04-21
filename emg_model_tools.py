"""Signal processing, feature extraction, training, and live inference.

This module is the shared contract between offline model training and live
prediction. Both paths should use the same baseline-centering, event detection,
and feature-extraction logic so that training-time feature vectors match the
feature vectors produced by ``LivePredictor`` at runtime.

Saved model bundles are plain dictionaries persisted with ``joblib``. Each
bundle contains the fitted classifier, fitted ``StandardScaler``, feature names,
selected channels, training report, and live-prediction config.
"""

import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emg_data_tools import (
    CHANNEL_NAMES,
    DEFAULT_CHANNEL,
    DB_ROOT,
    load_calibration,
    list_capture_groups,
    list_data_files,
    normalize_channel_selection,
    sanitize_label_name,
    sanitize_user_name,
)


MODEL_ROOT = DB_ROOT / "models"
PROCESSED_PREVIEW_ROOT = DB_ROOT / "processed_previews"
LEGACY_FEATURE_NAMES = ("mav", "rms", "var", "wl", "zc")
V3_FEATURE_NAMES = (
    "mav",
    "rms",
    "var",
    "wl",
    "zc",
    "ssc",
    "wamp",
    "iav",
    "ld",
    "damv",
    "dasdv",
    "dvarv",
    "msr",
    "myop",
    "peak",
    "ptp",
    "duration",
    "ar1",
    "ar2",
    "ar3",
    "ar4",
)
# Token-based burst-shape features (aggregated over N equal-sized sub-windows
# inside the event slice). Helps separate transient bursts (snap) from
# sustained contractions (fist) by encoding WHERE in the burst the peak sits
# and HOW concentrated the activity is.
TOKEN_FEATURE_NAMES = (
    "tk_peak_pos",     # normalized argmax of per-token peak, 0..1
    "tk_mav_max",      # strongest sub-window MAV
    "tk_mav_min",      # weakest sub-window MAV
    "tk_mav_std",      # std of sub-window MAVs
    "tk_active_ratio", # fraction of tokens with MAV >= 0.5 * tk_mav_max
)
TOKEN_COUNT = 5

SINGLE_FEATURE_NAMES = V3_FEATURE_NAMES + TOKEN_FEATURE_NAMES
FEATURE_NAMES = SINGLE_FEATURE_NAMES
CURRENT_PREPROCESSING_VERSION = 4
MODEL_TYPE_CHOICES = (
    ("auto_best", "Auto Best"),
    ("svm", "SVM"),
    ("random_forest", "Random Forest"),
    ("knn", "KNN"),
    ("logistic_regression", "Logistic Regression"),
    ("mlp", "MLP"),
)
MODEL_TYPE_DISPLAY_BY_KEY = dict(MODEL_TYPE_CHOICES)
MODEL_TYPE_ALIASES = {
    key: key
    for key, _display_name in MODEL_TYPE_CHOICES
}
MODEL_TYPE_ALIASES.update({
    display_name.lower().replace(" ", "_"): key
    for key, display_name in MODEL_TYPE_CHOICES
})
MODEL_TYPE_ALIASES.update({
    "auto": "auto_best",
    "best": "auto_best",
    "auto_select": "auto_best",
    "slp": "mlp",
    "single_layer_perceptron": "mlp",
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
})
AUTO_MODEL_TYPE_CANDIDATES = (
    "svm",
    "random_forest",
    "knn",
    "logistic_regression",
    "mlp",
)
AUTO_MODEL_SELECTION_SPLITS = 5

DEFAULT_FALLBACK_EVENT_THRESHOLD = 0.02
DEFAULT_MIN_NOISE_STD = 0.005
# Calibration is considered broken when its reported noise std is larger than
# this. Typical resting EMG noise is 1-15 mV; anything well above that means
# the calibration was captured while the user was moving (gestures got
# averaged in) and the resulting threshold would be absurdly high, preventing
# any real gesture from triggering the event detector. When we see this, we
# treat the calibration as if it did not exist.
MAX_REASONABLE_NOISE_STD = 0.05
DEFAULT_EVENT_MIN_SECONDS = 0.08
DEFAULT_EVENT_PAD_SECONDS = 0.04
DEFAULT_EVENT_QUIET_SECONDS = 0.20
DEFAULT_EVENT_COMPLETION_MAX_LAG_SECONDS = 0.35
DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS = 1.0
# During the quiet-tail check in LivePredictor, the event is considered "still
# ongoing" only if at least this fraction of the tail samples are above the
# activity threshold. Prevents isolated noise spikes from blocking prediction
# (the previous "any sample" check caused the predictor to get stuck on
# 'capturing' forever when hardware noise occasionally crossed the threshold).
DEFAULT_EVENT_TAIL_ACTIVE_FRACTION = 0.40


def sanitize_model_name(raw_name):
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in raw_name.strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or f"model_{time.strftime('%Y%m%d_%H%M%S')}"


def sanitize_model_type(raw_model_type):
    normalized = (raw_model_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = MODEL_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_TYPE_DISPLAY_BY_KEY:
        supported_types = ", ".join(MODEL_TYPE_DISPLAY_BY_KEY.values())
        raise ValueError(f"Unsupported model type: {raw_model_type}. Choose one of: {supported_types}.")
    return normalized


def model_type_display_name(raw_model_type):
    return MODEL_TYPE_DISPLAY_BY_KEY[sanitize_model_type(raw_model_type)]


def model_type_labels():
    return [display_name for _key, display_name in MODEL_TYPE_CHOICES]


def feature_names_for_version(preprocessing_version):
    if preprocessing_version is None:
        return LEGACY_FEATURE_NAMES
    version = int(preprocessing_version)
    if version < 3:
        return LEGACY_FEATURE_NAMES
    if version == 3:
        return V3_FEATURE_NAMES
    return FEATURE_NAMES


def feature_names_for_bundle(bundle):
    feature_names = bundle.get("feature_names")
    if feature_names:
        return tuple(feature_names)
    return feature_names_for_version(bundle.get("preprocessing_version"))


def _build_classifier(mode, model_type, train_size=None):
    safe_model_type = sanitize_model_type(model_type)
    effective_train_size = max(1, int(train_size or 1))

    if safe_model_type == "auto_best":
        raise ValueError("auto_best is a selection strategy, not a classifier.")
    if safe_model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        )
    elif safe_model_type == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=42,
        )
    elif safe_model_type == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=min(5, effective_train_size),
            weights="distance",
        )
    elif safe_model_type == "svm":
        classifier = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )
    elif safe_model_type == "logistic_regression":
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=1200,
            C=1.0,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return classifier, safe_model_type, model_type_display_name(safe_model_type)


def normalize_model_channels(selected_channels=None, channel=DEFAULT_CHANNEL):
    channels = normalize_channel_selection(selected_channels=selected_channels, channel=channel)
    if not channels:
        channels = [DEFAULT_CHANNEL]
    return channels, "_".join(channels)


def parse_channel_group(channel_group):
    if isinstance(channel_group, str) and channel_group in CHANNEL_NAMES:
        return [channel_group]
    return normalize_channel_selection(selected_channels=(channel_group or "").split("_"))


def model_dir_for_mode(mode, channel=DEFAULT_CHANNEL, selected_channels=None):
    _channels, channel_key = normalize_model_channels(selected_channels=selected_channels, channel=channel)
    return MODEL_ROOT / mode / channel_key


def model_path_for_name(mode, model_name, channel=DEFAULT_CHANNEL, selected_channels=None):
    return model_dir_for_mode(mode, channel=channel, selected_channels=selected_channels) / f"{sanitize_model_name(model_name)}.joblib"


def _model_confusion_heatmap_path(model_path, model_type=None):
    suffix = "_confusion_matrix"
    if model_type:
        suffix = f"_{sanitize_model_type(model_type)}{suffix}"
    return model_path.with_name(f"{model_path.stem}{suffix}.png")


def _clear_model_confusion_heatmaps(model_path):
    heatmap_paths = [model_path.with_name(f"{model_path.stem}_confusion_matrix.png")]
    heatmap_paths.extend(model_path.parent.glob(f"{model_path.stem}_*_confusion_matrix.png"))
    for heatmap_path in heatmap_paths:
        try:
            if heatmap_path.exists():
                heatmap_path.unlink()
        except OSError:
            pass


def _matching_model_paths(mode, channel=DEFAULT_CHANNEL, selected_channels=None):
    requested_channels, requested_key = normalize_model_channels(selected_channels=selected_channels, channel=channel)
    requested_set = set(requested_channels)
    mode_root = MODEL_ROOT / mode
    matching_paths = []

    if not mode_root.exists():
        return []

    for child in sorted(mode_root.iterdir()):
        if child.is_dir():
            try:
                child_channels = set(parse_channel_group(child.name))
            except ValueError:
                continue
            if child_channels and requested_set.intersection(child_channels):
                matching_paths.extend(sorted(child.glob("*.joblib")))
        elif child.suffix == ".joblib" and requested_key == DEFAULT_CHANNEL:
            matching_paths.append(child)

    return matching_paths


def list_saved_models(mode, channel=DEFAULT_CHANNEL, selected_channels=None):
    model_names = set()

    for model_path in _matching_model_paths(mode, channel=channel, selected_channels=selected_channels):
        model_names.add(model_path.stem)

    return sorted(model_names)


def delete_named_model(mode, model_name):
    """Delete all model artifacts matching *model_name* across every channel directory for *mode*.

    Returns the number of .joblib bundles deleted.
    """
    safe_name = sanitize_model_name(model_name)
    mode_root = MODEL_ROOT / mode
    deleted = 0
    if not mode_root.exists():
        return deleted
    # Search all channel sub-directories and the mode root itself
    for candidate in mode_root.rglob(f"{safe_name}.joblib"):
        _clear_model_confusion_heatmaps(candidate)
        candidate.unlink()
        deleted += 1
    return deleted


def load_named_model(mode, model_name, channel=DEFAULT_CHANNEL, selected_channels=None):
    model_path = model_path_for_name(mode, model_name, channel=channel, selected_channels=selected_channels)
    if model_path.exists():
        return joblib.load(model_path)

    safe_model_name = sanitize_model_name(model_name)
    for candidate in _matching_model_paths(mode, channel=channel, selected_channels=selected_channels):
        if candidate.stem == safe_model_name:
            return joblib.load(candidate)

    return joblib.load(model_path)


def _estimate_sampling_rate(time_s, default_fs=200.0):
    # Use float64 — Arduino/Python wallclock timestamps are large (~1.7e9)
    # and float32 silently rounds adjacent samples to the same value,
    # yielding dt=0 and forcing this function to fall back to default_fs.
    time_s = np.asarray(time_s, dtype=np.float64)
    if len(time_s) < 2:
        return float(default_fs)

    dt = float(np.mean(np.diff(time_s)))
    return float(1.0 / dt) if dt > 0 else float(default_fs)


def _event_sample_counts(fs, min_seconds=DEFAULT_EVENT_MIN_SECONDS, pad_seconds=DEFAULT_EVENT_PAD_SECONDS):
    min_samples = max(int(round(fs * min_seconds)), 8)
    pad_samples = max(int(round(fs * pad_seconds)), 2)
    return min_samples, pad_samples


def _channel_calibration_is_sane(channel_calibration):
    """Return True when this per-channel calibration looks like it came from
    actual rest (voltage_std within a believable range). A huge std is almost
    always a sign that Calibrate Idle was clicked while the user was moving."""
    if channel_calibration is None:
        return False
    std = float(channel_calibration.get("voltage_std", 0.0))
    if std <= 0 or std > MAX_REASONABLE_NOISE_STD:
        return False
    return True


def _baseline_mean(channel_calibration, signal):
    # ALWAYS estimate the baseline from the current signal (per-window
    # 10th percentile), NEVER from a global calibration's voltage_mean.
    #
    # Rationale: the electrode DC baseline drifts between the moment a
    # capture was recorded and the moment the model is trained or tested
    # (different session, different skin/contact state, sometimes days
    # apart). Using a global voltage_mean causes the training-time event
    # slice to be wildly different from the live-time event slice, which
    # looks like "training F1 ~ 1 but live accuracy terrible" — the two
    # pipelines produce totally different feature distributions.
    #
    # The per-signal 10th percentile adapts to whatever rest level the
    # current window has, so training features (extracted per-capture)
    # and live features (extracted per-buffer) end up consistent.
    if signal is not None and len(signal) > 0:
        return float(np.percentile(np.asarray(signal, dtype=np.float64), 10.0))
    if _channel_calibration_is_sane(channel_calibration):
        return float(channel_calibration.get("voltage_mean", 0.0))
    return 0.0


def _noise_std(channel_calibration):
    if not _channel_calibration_is_sane(channel_calibration):
        return None
    return max(float(channel_calibration.get("voltage_std", 0.0)), DEFAULT_MIN_NOISE_STD)


def center_signal(signal, channel_calibration=None):
    signal = np.asarray(signal, dtype=np.float32)
    return signal - _baseline_mean(channel_calibration, signal)


def _smooth_envelope(centered_signal):
    centered_signal = np.asarray(centered_signal, dtype=np.float32)
    if len(centered_signal) < 3:
        return np.abs(centered_signal)

    win = min(11, max(3, len(centered_signal) // 50))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(np.abs(centered_signal), kernel, mode="same")


def _event_threshold(channel_calibration, k=3.0, fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD):
    noise_std = _noise_std(channel_calibration)
    if noise_std is None:
        return float(fallback_threshold)
    return max(float(k) * noise_std, float(fallback_threshold))


def _largest_active_slice(event_mask, min_samples=20, pad_samples=0):
    indices = np.flatnonzero(event_mask)
    if len(indices) < min_samples:
        return None

    best_start = best_end = None
    run_start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx != prev + 1:
            if best_start is None or (prev - run_start) > (best_end - best_start):
                best_start, best_end = run_start, prev
            run_start = idx
        prev = idx

    if best_start is None or (prev - run_start) > (best_end - best_start):
        best_start, best_end = run_start, prev

    if best_start is None:
        return None

    start = max(0, best_start - pad_samples)
    end = min(len(event_mask), best_end + pad_samples + 1)
    if end - start < min_samples:
        return None
    return slice(start, end)


def _latest_active_slice(event_mask, min_samples=20, pad_samples=0):
    indices = np.flatnonzero(event_mask)
    if len(indices) < min_samples:
        return None

    run_start = indices[0]
    prev = indices[0]
    best_start = best_end = None

    for idx in indices[1:]:
        if idx != prev + 1:
            if prev - run_start + 1 >= min_samples:
                best_start, best_end = run_start, prev
            run_start = idx
        prev = idx

    if prev - run_start + 1 >= min_samples:
        best_start, best_end = run_start, prev

    if best_start is None:
        return None

    start = max(0, best_start - pad_samples)
    end = min(len(event_mask), best_end + pad_samples + 1)
    if end - start < min_samples:
        return None
    return slice(start, end)


def _event_mask_for_signal(signal, channel_calibration=None, k=3.0,
                           fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD):
    centered_signal = center_signal(signal, channel_calibration)
    threshold = _event_threshold(channel_calibration, k=k, fallback_threshold=fallback_threshold)
    return centered_signal, _smooth_envelope(centered_signal) > threshold


def _analyze_feature_window(signals, channel_calibrations=None, k=3.0,
                            min_samples=20, pad_samples=0,
                            fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                            slice_strategy="largest"):
    if not signals:
        return [], np.zeros(0, dtype=bool), None

    min_len = min(len(signal) for signal in signals)
    if min_len < 2:
        centered = [center_signal(signal, None)[:min_len] for signal in signals]
        return centered, np.zeros(min_len, dtype=bool), None

    channel_calibrations = list(channel_calibrations or [])
    if len(channel_calibrations) < len(signals):
        channel_calibrations.extend([None] * (len(signals) - len(channel_calibrations)))

    centered_signals = []
    combined_mask = np.zeros(min_len, dtype=bool)

    for signal, channel_calibration in zip(signals, channel_calibrations):
        raw_signal = np.asarray(signal, dtype=np.float32)[:min_len]
        centered_signal, event_mask = _event_mask_for_signal(
            raw_signal,
            channel_calibration=channel_calibration,
            k=k,
            fallback_threshold=fallback_threshold,
        )
        centered_signals.append(centered_signal)
        combined_mask |= event_mask

    if slice_strategy == "latest":
        event_slice = _latest_active_slice(combined_mask, min_samples=min_samples, pad_samples=pad_samples)
    else:
        event_slice = _largest_active_slice(combined_mask, min_samples=min_samples, pad_samples=pad_samples)
    return centered_signals, combined_mask, event_slice


def prepare_feature_signals(signals, channel_calibrations=None, k=3.0,
                            min_samples=20, pad_samples=0,
                            fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                            keep_full_window=False):
    centered_signals, _combined_mask, event_slice = _analyze_feature_window(
        signals,
        channel_calibrations=channel_calibrations,
        k=k,
        min_samples=min_samples,
        pad_samples=pad_samples,
        fallback_threshold=fallback_threshold,
        slice_strategy="largest",
    )
    if event_slice is None:
        return centered_signals if keep_full_window else centered_signals, False

    return [signal[event_slice] for signal in centered_signals], True


def prepare_model_signals(signals, channel_calibrations=None, fs=200.0, label=None,
                          k=3.0, fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                          keep_full_window=None):
    """Prepare raw capture signals for feature extraction.

    Gesture labels are trimmed to the detected active event slice. ``rest``
    captures keep the full window unless explicitly overridden because rest
    should not contain an active event. Returns ``(prepared_signals, has_event)``
    or ``(None, False)`` when a non-rest capture has no usable event.
    """
    min_samples, pad_samples = _event_sample_counts(fs)
    if keep_full_window is None:
        keep_full_window = label == "rest"

    prepared_signals, has_event = prepare_feature_signals(
        signals,
        channel_calibrations=channel_calibrations,
        k=k,
        min_samples=min_samples,
        pad_samples=pad_samples,
        fallback_threshold=fallback_threshold,
        keep_full_window=keep_full_window,
    )

    if label not in (None, "rest") and not keep_full_window and not has_event:
        return None, False

    return prepared_signals, has_event


def analyze_model_window(signals, channel_calibrations=None, fs=200.0, k=3.0,
                         fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                         slice_strategy="largest"):
    """Analyze a live/training window and return centered signals plus event slice."""
    min_samples, pad_samples = _event_sample_counts(fs)
    return _analyze_feature_window(
        signals,
        channel_calibrations=channel_calibrations,
        k=k,
        min_samples=min_samples,
        pad_samples=pad_samples,
        fallback_threshold=fallback_threshold,
        slice_strategy=slice_strategy,
    )


def _zero_crossing_count(signal, noise_std=None):
    if len(signal) < 2:
        return 0.0

    threshold = max((noise_std or 0.0) * 0.5, 1e-3)
    sign_change = signal[:-1] * signal[1:] < 0
    magnitude_change = np.abs(signal[:-1] - signal[1:]) >= threshold
    return float(np.sum(sign_change & magnitude_change))


def _slope_sign_change_count(signal, noise_std=None):
    if len(signal) < 3:
        return 0.0

    threshold = max((noise_std or 0.0) * 0.5, 1e-3)
    diffs = np.diff(signal)
    sign_change = diffs[:-1] * diffs[1:] < 0
    magnitude_change = np.maximum(np.abs(diffs[:-1]), np.abs(diffs[1:])) >= threshold
    return float(np.sum(sign_change & magnitude_change))


def _willison_amplitude_count(signal, noise_std=None):
    if len(signal) < 2:
        return 0.0

    threshold = max((noise_std or 0.0) * 0.5, 1e-3)
    return float(np.sum(np.abs(np.diff(signal)) >= threshold))


def _log_detector(signal):
    if len(signal) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(np.abs(signal) + 1e-6))))


def _autoregressive_coefficients(signal, order=4):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) <= order:
        return np.zeros(order, dtype=np.float32)

    y = signal[order:]
    x = np.column_stack([signal[order - lag - 1: -lag - 1] for lag in range(order)])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if len(coeffs) < order:
        coeffs = np.pad(coeffs, (0, order - len(coeffs)))
    return coeffs[:order]


def _token_statistics(signal, n_tokens=TOKEN_COUNT):
    """Split a signal into `n_tokens` equal chunks and return per-token MAV and peak.

    Empty chunks (when the signal is shorter than `n_tokens`) contribute zeros
    and leave the output length at `n_tokens` so downstream aggregation stays
    consistent.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == 0 or n_tokens < 1:
        return None

    chunks = np.array_split(signal, n_tokens)
    mavs = np.array(
        [float(np.mean(np.abs(chunk))) if len(chunk) > 0 else 0.0 for chunk in chunks],
        dtype=np.float32,
    )
    peaks = np.array(
        [float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0 for chunk in chunks],
        dtype=np.float32,
    )
    return {"mav": mavs, "peak": peaks}


def extract_feature_vector(signal, noise_std=None, feature_names=None):
    """Convert one prepared EMG signal into a fixed-length feature vector.

    The input may be a variable-length event slice. The output order follows
    ``feature_names`` and is stable across training and live prediction.
    """
    signal = np.asarray(signal, dtype=np.float32)
    feature_names = tuple(feature_names or FEATURE_NAMES)

    if len(signal) == 0:
        return np.zeros(len(feature_names), dtype=np.float32)

    abs_signal = np.abs(signal)
    diff_signal = np.diff(signal)
    ar_coeffs = None
    token_stats = None
    values = []

    for name in feature_names:
        if name == "mav":
            values.append(float(np.mean(abs_signal)))
        elif name == "rms":
            values.append(float(np.sqrt(np.mean(signal ** 2))))
        elif name == "var":
            values.append(float(np.var(signal)))
        elif name == "wl":
            values.append(float(np.sum(np.abs(diff_signal))))
        elif name == "zc":
            values.append(_zero_crossing_count(signal, noise_std=noise_std))
        elif name == "ssc":
            values.append(_slope_sign_change_count(signal, noise_std=noise_std))
        elif name == "wamp":
            values.append(_willison_amplitude_count(signal, noise_std=noise_std))
        elif name == "iav":
            values.append(float(np.sum(abs_signal)))
        elif name == "ld":
            values.append(_log_detector(signal))
        elif name == "damv":
            values.append(float(np.mean(np.abs(diff_signal))) if len(diff_signal) else 0.0)
        elif name == "dasdv":
            values.append(float(np.sqrt(np.mean(diff_signal ** 2))) if len(diff_signal) else 0.0)
        elif name == "dvarv":
            values.append(float(np.var(diff_signal)) if len(diff_signal) else 0.0)
        elif name == "msr":
            values.append(float(np.mean(np.sqrt(abs_signal + 1e-6)) ** 2))
        elif name == "myop":
            threshold = max((noise_std or 0.0) * 0.5, float(np.std(signal) * 0.5), 1e-3)
            values.append(float(np.mean(abs_signal >= threshold)))
        elif name == "peak":
            values.append(float(np.max(abs_signal)))
        elif name == "ptp":
            values.append(float(np.ptp(signal)))
        elif name == "duration":
            values.append(float(len(signal)))
        elif name.startswith("ar") and name[2:].isdigit():
            if ar_coeffs is None:
                max_order = max(int(feature_name[2:]) for feature_name in feature_names if feature_name.startswith("ar") and feature_name[2:].isdigit())
                ar_coeffs = _autoregressive_coefficients(signal, order=max_order)
            index = int(name[2:]) - 1
            values.append(float(ar_coeffs[index]) if index < len(ar_coeffs) else 0.0)
        elif name.startswith("tk_"):
            if token_stats is None:
                token_stats = _token_statistics(signal, n_tokens=TOKEN_COUNT)
            if token_stats is None:
                values.append(0.0)
                continue
            tk_mav = token_stats["mav"]
            tk_peak = token_stats["peak"]
            tk_n = len(tk_peak)
            if name == "tk_peak_pos":
                if tk_n <= 1:
                    values.append(0.0)
                else:
                    values.append(float(np.argmax(tk_peak)) / float(tk_n - 1))
            elif name == "tk_mav_max":
                values.append(float(np.max(tk_mav)))
            elif name == "tk_mav_min":
                values.append(float(np.min(tk_mav)))
            elif name == "tk_mav_std":
                values.append(float(np.std(tk_mav)))
            elif name == "tk_active_ratio":
                mav_max = float(np.max(tk_mav))
                if mav_max <= 1e-6:
                    values.append(0.0)
                else:
                    values.append(float(np.mean(tk_mav >= 0.5 * mav_max)))
            else:
                raise ValueError(f"Unknown token feature name: {name}")
        else:
            raise ValueError(f"Unknown feature name: {name}")

    return np.asarray(values, dtype=np.float32)


def extract_single_features(signal, noise_std=None, feature_names=None):
    return extract_feature_vector(signal, noise_std=noise_std, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Legacy Conv1D compatibility
# ---------------------------------------------------------------------------

class EMGConv1DNet:
    """Compatibility placeholder so legacy Conv1D bundles can still be unpickled."""


class Conv1DClassifier:
    """Compatibility placeholder for retired Conv1D models."""

    def __init__(self, *_args, **_kwargs):
        raise ValueError(
            "Conv1D models have been retired from this project. "
            "Train a feature-based model instead."
        )

    def fit(self, *_args, **_kwargs):
        raise ValueError("Conv1D training is no longer supported.")

    def predict(self, *_args, **_kwargs):
        raise ValueError("Legacy Conv1D models are no longer supported.")

    def predict_proba(self, *_args, **_kwargs):
        raise ValueError("Legacy Conv1D models are no longer supported.")

    def fine_tune(self, *_args, **_kwargs):
        raise ValueError("Conv1D fine-tuning is no longer supported.")


def _load_channel_capture(csv_file):
    df = pd.read_csv(csv_file)
    if len(df) < 10:
        return None

    return {
        "label": df["label"].iloc[0],
        "time_s": df["time_s"].values.astype(np.float32),
        "signal": df["voltage"].values.astype(np.float32),
    }


def _load_group_capture(record, selected_channels):
    channel_payloads = {}
    label = None

    for channel in selected_channels:
        csv_file = record["files_by_channel"].get(channel)
        if csv_file is None:
            return None

        payload = _load_channel_capture(csv_file)
        if payload is None:
            return None

        if label is None:
            label = payload["label"]
        elif payload["label"] != label:
            raise ValueError(f"Mismatched labels inside capture group {record['filename']}.")

        channel_payloads[channel] = payload

    return channel_payloads, label


def make_windows(signal, fs, window_seconds, stride_seconds):
    window_size = int(window_seconds * fs)
    stride_size = int(stride_seconds * fs)

    if window_size < 1 or stride_size < 1:
        return []

    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start:start + window_size])
        start += stride_size

    return windows


def _safe_split(X, y, test_size=0.2, groups=None, random_state=42):
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Need at least two different labels to train a model.")

    if len(X) < 6:
        raise ValueError("Need more samples before training.")

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            y_train = y[train_idx]
            y_test = y[test_idx]
            if len(np.unique(y_train)) >= 2 and len(y_test) > 0:
                return X[train_idx], X[test_idx], y_train, y_test

    stratify = y if np.min(counts) >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def _format_confusion_matrix(labels, matrix, title=None, note=None):
    label_names = [str(label) for label in labels]
    matrix = np.asarray(matrix, dtype=int)
    if matrix.size == 0 or not label_names:
        return "(confusion matrix unavailable)"

    row_header = "actual/pred"
    row_width = max(len(row_header), *(len(label) for label in label_names))
    col_widths = []
    for col_idx, label in enumerate(label_names):
        values = [str(int(matrix[row_idx, col_idx])) for row_idx in range(len(label_names))]
        col_widths.append(max(len(label), *(len(value) for value in values)))

    header = (
        f"{row_header:<{row_width}}  "
        + "  ".join(f"{label:>{col_widths[idx]}}" for idx, label in enumerate(label_names))
    )
    lines = []
    if title:
        lines.append(title)
    if note:
        lines.append(note)
    lines.extend([
        "rows=actual, columns=predicted",
        header,
        "-" * len(header),
    ])
    for row_idx, label in enumerate(label_names):
        values = "  ".join(
            f"{int(matrix[row_idx, col_idx]):>{col_widths[col_idx]}}"
            for col_idx in range(len(label_names))
        )
        lines.append(f"{label:<{row_width}}  {values}")
    return "\n".join(lines)


def _save_confusion_matrix_heatmap(labels, matrix, output_path, title=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    label_names = [str(label) for label in labels]
    matrix = np.asarray(matrix, dtype=int)
    if matrix.size == 0 or not label_names:
        raise ValueError("Cannot save an empty confusion matrix heatmap.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis_size = max(4.0, min(12.0, 2.0 + 0.65 * len(label_names)))
    fig = Figure(figsize=(axis_size, axis_size), dpi=140, constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(label_names))
    ax.set(
        xticks=tick_positions,
        yticks=tick_positions,
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=title or "Confusion Matrix",
    )
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    max_value = int(matrix.max()) if matrix.size else 0
    threshold = max_value / 2.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def _save_training_confusion_heatmaps(bundle, model_path):
    _clear_model_confusion_heatmaps(model_path)

    heatmap_path = _model_confusion_heatmap_path(model_path)
    _save_confusion_matrix_heatmap(
        bundle["confusion_labels"],
        bundle["confusion_matrix"],
        heatmap_path,
        title=f"{bundle.get('model_type_display', 'Model')} Confusion Matrix",
    )
    bundle["confusion_heatmap_path"] = str(heatmap_path)

    benchmark_heatmap_paths = []
    for result in bundle.get("benchmark_results") or []:
        candidate_path = _model_confusion_heatmap_path(model_path, result["model_type"])
        _save_confusion_matrix_heatmap(
            result["confusion_labels"],
            result["confusion_matrix"],
            candidate_path,
            title=f"{result.get('model_type_display', result['model_type'])} Confusion Matrix",
        )
        result["confusion_heatmap_path"] = str(candidate_path)
        benchmark_heatmap_paths.append({
            "model_type": result["model_type"],
            "model_type_display": result.get("model_type_display", result["model_type"]),
            "path": str(candidate_path),
        })

    bundle["benchmark_confusion_heatmap_paths"] = benchmark_heatmap_paths
    return heatmap_path


def _evaluate_split(mode, model_type, X_train, X_test, y_train, y_test, labels=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier, safe_model_type, display_model_type = _build_classifier(
        mode,
        model_type,
        train_size=len(X_train_scaled),
    )
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    return {
        "model_type": safe_model_type,
        "model_type_display": display_model_type,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_test, y_pred, labels=labels, zero_division=0),
        "confusion_labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_text": _format_confusion_matrix(labels, matrix),
        "_y_true": y_test.tolist(),
        "_y_pred": y_pred.tolist(),
    }


def _benchmark_model_type(mode, model_type, X, y, groups=None, repeats=AUTO_MODEL_SELECTION_SPLITS):
    safe_model_type = sanitize_model_type(model_type)
    labels = sorted(np.unique(y).tolist())
    split_metrics = []

    for split_idx in range(max(1, int(repeats))):
        try:
            X_train, X_test, y_train, y_test = _safe_split(
                X,
                y,
                groups=groups,
                random_state=42 + split_idx,
            )
        except ValueError:
            continue

        metrics = _evaluate_split(mode, safe_model_type, X_train, X_test, y_train, y_test, labels=labels)
        metrics["split_index"] = split_idx
        split_metrics.append(metrics)

    if not split_metrics:
        raise ValueError(f"Unable to evaluate model type: {model_type}")

    aggregate_matrix = np.sum(
        [np.asarray(item["confusion_matrix"], dtype=int) for item in split_metrics],
        axis=0,
    )
    aggregate_y_true = np.concatenate([np.asarray(item["_y_true"]) for item in split_metrics])
    aggregate_y_pred = np.concatenate([np.asarray(item["_y_pred"]) for item in split_metrics])

    return {
        "model_type": safe_model_type,
        "model_type_display": model_type_display_name(safe_model_type),
        "accuracy_mean": float(np.mean([item["accuracy"] for item in split_metrics])),
        "macro_f1_mean": float(np.mean([item["macro_f1"] for item in split_metrics])),
        "split_count": len(split_metrics),
        "report": classification_report(aggregate_y_true, aggregate_y_pred, labels=labels, zero_division=0),
        "confusion_labels": labels,
        "confusion_matrix": aggregate_matrix.tolist(),
        "confusion_matrix_text": _format_confusion_matrix(
            labels,
            aggregate_matrix,
            title=f"{model_type_display_name(safe_model_type)} confusion matrix",
            note="summed across repeated holdout splits",
        ),
    }


def _format_benchmark_summary(benchmark_results, selected_model_type):
    lines = [
        "Auto model selection (avg repeated holdout)",
        "-" * 40,
        "Winner marked with *",
    ]
    for result in benchmark_results:
        marker = "*" if result["model_type"] == selected_model_type else " "
        lines.append(
            f"{marker} {result['model_type_display']:<19} "
            f"macro-F1={result['macro_f1_mean']:.3f}  "
            f"acc={result['accuracy_mean']:.3f}  "
            f"splits={result['split_count']}"
        )
    return "\n".join(lines)


def _format_benchmark_confusion_matrices(benchmark_results):
    if not benchmark_results:
        return None
    lines = [
        "Per-model confusion matrices",
        "-" * 40,
        "Auto Best matrices are summed across repeated holdout splits.",
    ]
    for result in benchmark_results:
        matrix_text = result.get("confusion_matrix_text")
        if matrix_text:
            lines.extend(["", matrix_text])
    return "\n".join(lines)


def _source_user_name(source):
    try:
        if isinstance(source, Path):
            csv_file = source
        else:
            csv_file = next(iter(source["files_by_channel"].values()))
        rel_path = csv_file.relative_to(DB_ROOT / "users")
        return rel_path.parts[0]
    except (KeyError, StopIteration, TypeError, ValueError):
        return None


def _resolve_sample_calibration(calibration, source):
    if calibration is None:
        return None
    if isinstance(calibration, dict) and "channels" in calibration:
        return calibration

    user_name = _source_user_name(source)
    if user_name is None or not isinstance(calibration, dict):
        return None

    user_calibration = calibration.get(user_name)
    if isinstance(user_calibration, dict) and "channels" in user_calibration:
        return user_calibration
    return None


def trim_to_events(signal, channel_calibration, k=3.0, min_samples=20):
    prepared_signals, has_event = prepare_feature_signals(
        [signal],
        channel_calibrations=[channel_calibration],
        k=k,
        min_samples=min_samples,
        fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
        keep_full_window=True,
    )
    if not prepared_signals:
        return np.asarray(signal, dtype=np.float32)
    return prepared_signals[0] if has_event else center_signal(signal, channel_calibration)


def is_event_window(signal, channel_calibration, k=3.0):
    centered_signal, event_mask = _event_mask_for_signal(
        signal,
        channel_calibration=channel_calibration,
        k=k,
        fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
    )
    min_samples = max(len(centered_signal) // 8, 4)
    return _largest_active_slice(event_mask, min_samples=min_samples, pad_samples=0) is not None


def _record_channels(record):
    if not isinstance(record, dict):
        return []

    files_by_channel = record.get("files_by_channel", {})
    channels = [ch for ch in CHANNEL_NAMES if ch in files_by_channel]
    if channels:
        return channels

    listed_channels = record.get("channels", [])
    if listed_channels:
        return normalize_channel_selection(selected_channels=listed_channels)
    return []


def prepare_capture_preview(record, calibration=None, k=None):
    """Build a preview of the exact centered/trimmed data used for feature extraction."""
    channels = _record_channels(record)
    if not channels:
        raise ValueError("Capture record does not contain any readable channels.")

    mode = record.get("mode", "single")
    sample_calibration = _resolve_sample_calibration(calibration, record)
    cal_channels = (sample_calibration or {}).get("channels", {})
    event_k = float((sample_calibration or {}).get("event_threshold_k", 3.0)) if k is None else float(k)

    if len(channels) == 1:
        channel = channels[0]
        payload = _load_channel_capture(record["files_by_channel"][channel])
        if payload is None:
            raise ValueError(f"Capture '{record.get('filename', '')}' is too short to preview.")
        label = payload["label"]
        aligned_time = np.asarray(payload["time_s"], dtype=np.float32)
        raw_signals = [np.asarray(payload["signal"], dtype=np.float32)]
    else:
        loaded_group = _load_group_capture(record, channels)
        if loaded_group is None:
            raise ValueError(f"Capture '{record.get('filename', '')}' is incomplete or too short to preview.")
        channel_payloads, label = loaded_group
        min_len = min(len(channel_payloads[ch]["signal"]) for ch in channels)
        aligned_time = np.asarray(channel_payloads[channels[0]]["time_s"][:min_len], dtype=np.float32)
        raw_signals = [
            np.asarray(channel_payloads[ch]["signal"][:min_len], dtype=np.float32)
            for ch in channels
        ]

    min_len = min(len(signal) for signal in raw_signals)
    aligned_time = aligned_time[:min_len]
    raw_signals = [signal[:min_len] for signal in raw_signals]
    channel_calibrations = [cal_channels.get(ch) for ch in channels]

    centered_signals = []
    envelopes = []
    event_masks = []
    thresholds = []
    baselines = []
    noise_stds = []
    combined_mask = np.zeros(min_len, dtype=bool)

    for raw_signal, channel_calibration in zip(raw_signals, channel_calibrations):
        centered_signal = center_signal(raw_signal, channel_calibration)
        envelope = _smooth_envelope(centered_signal)
        threshold = _event_threshold(
            channel_calibration,
            k=event_k,
            fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
        )
        event_mask = envelope > threshold

        centered_signals.append(centered_signal)
        envelopes.append(envelope)
        event_masks.append(event_mask)
        thresholds.append(float(threshold))
        baselines.append(float(_baseline_mean(channel_calibration, raw_signal)))
        noise_stds.append(_noise_std(channel_calibration))
        combined_mask |= event_mask

    fs = _estimate_sampling_rate(aligned_time)
    min_samples, pad_samples = _event_sample_counts(fs)
    event_slice = _largest_active_slice(combined_mask, min_samples=min_samples, pad_samples=pad_samples)
    keep_full_window = label == "rest"

    if event_slice is not None:
        processed_slice = event_slice
        preview_status = "active_segment"
    elif keep_full_window:
        processed_slice = slice(0, min_len)
        preview_status = "full_window"
    else:
        processed_slice = None
        preview_status = "skipped_no_event"

    if processed_slice is None:
        processed_original_time_s = np.asarray([], dtype=np.float32)
        processed_time_s = np.asarray([], dtype=np.float32)
    else:
        processed_original_time_s = aligned_time[processed_slice]
        processed_time_s = processed_original_time_s - processed_original_time_s[0]

    feature_extractor = extract_single_features
    preview_feature_names = SINGLE_FEATURE_NAMES
    channels_data = {}

    for idx, channel in enumerate(channels):
        if processed_slice is None:
            processed_voltage = np.asarray([], dtype=np.float32)
            processed_envelope = np.asarray([], dtype=np.float32)
            features = None
        else:
            processed_voltage = centered_signals[idx][processed_slice]
            processed_envelope = envelopes[idx][processed_slice]
            features = feature_extractor(
                processed_voltage,
                noise_std=noise_stds[idx],
                feature_names=preview_feature_names,
            ).astype(np.float32)

        channels_data[channel] = {
            "raw_voltage": raw_signals[idx],
            "centered_voltage": centered_signals[idx],
            "envelope": envelopes[idx],
            "event_mask": event_masks[idx],
            "processed_voltage": processed_voltage,
            "processed_envelope": processed_envelope,
            "baseline_mean": baselines[idx],
            "noise_std": noise_stds[idx],
            "threshold": thresholds[idx],
            "features": features,
        }

    return {
        "user": _source_user_name(record) or "unknown_user",
        "mode": mode,
        "label": label,
        "filename": record.get("filename", ""),
        "timestamp": record.get("timestamp", ""),
        "channels": channels,
        "fs": float(fs),
        "event_k": float(event_k),
        "status": preview_status,
        "has_event": event_slice is not None,
        "raw_sample_count": int(min_len),
        "processed_sample_count": int(len(processed_time_s)),
        "processed_slice_start": None if processed_slice is None else int(processed_slice.start),
        "processed_slice_stop": None if processed_slice is None else int(processed_slice.stop),
        "raw_time_s": aligned_time,
        "processed_time_s": processed_time_s,
        "processed_original_time_s": processed_original_time_s,
        "combined_event_mask": combined_mask,
        "feature_names": preview_feature_names,
        "channels_data": channels_data,
        "calibration_source": (sample_calibration or {}).get("source"),
    }


def export_processed_capture_preview(record, calibration=None, output_dir=None, k=None):
    preview = prepare_capture_preview(record, calibration=calibration, k=k)
    if preview["processed_sample_count"] < 1:
        raise ValueError("No processed segment was found for this capture. It would be skipped during training.")

    export_dir = Path(output_dir) if output_dir else PROCESSED_PREVIEW_ROOT / preview["user"] / preview["mode"]
    export_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(preview["filename"]).stem or sanitize_model_name(f"{preview['label']}_preview")
    processed_path = export_dir / f"{stem}_processed.csv"
    feature_path = export_dir / f"{stem}_features.csv"

    processed_columns = {
        "processed_time_s": preview["processed_time_s"],
        "original_time_s": preview["processed_original_time_s"],
    }
    for channel in preview["channels"]:
        channel_data = preview["channels_data"][channel]
        processed_columns[f"{channel}_centered_voltage"] = channel_data["processed_voltage"]
        processed_columns[f"{channel}_envelope"] = channel_data["processed_envelope"]
        processed_columns[f"{channel}_threshold"] = np.full(
            preview["processed_sample_count"],
            channel_data["threshold"],
            dtype=np.float32,
        )
    pd.DataFrame(processed_columns).to_csv(processed_path, index=False)

    feature_row = {
        "user": preview["user"],
        "mode": preview["mode"],
        "label": preview["label"],
        "filename": preview["filename"],
        "status": preview["status"],
        "raw_sample_count": preview["raw_sample_count"],
        "processed_sample_count": preview["processed_sample_count"],
        "fs": preview["fs"],
        "event_threshold_k": preview["event_k"],
    }
    for channel in preview["channels"]:
        channel_data = preview["channels_data"][channel]
        feature_row[f"{channel}_baseline_mean"] = channel_data["baseline_mean"]
        feature_row[f"{channel}_noise_std"] = channel_data["noise_std"]
        feature_row[f"{channel}_threshold"] = channel_data["threshold"]
        features = channel_data["features"]
        if features is not None:
            for name, value in zip(preview["feature_names"], features):
                feature_row[f"{channel}_{name}"] = float(value)
    pd.DataFrame([feature_row]).to_csv(feature_path, index=False)

    return {
        "preview": preview,
        "processed_csv": processed_path,
        "features_csv": feature_path,
    }


def _train_feature_bundle(mode, X, y, groups, dataset_source, model_type, feature_names, config):
    safe_requested_model_type = sanitize_model_type(model_type)
    all_labels = sorted(np.unique(y).tolist())
    benchmark_results = None
    benchmark_summary = None
    benchmark_confusion_matrices = None

    if safe_requested_model_type == "auto_best":
        benchmark_results = [
            _benchmark_model_type(mode, candidate_type, X, y, groups=groups)
            for candidate_type in AUTO_MODEL_TYPE_CANDIDATES
        ]
        benchmark_results.sort(
            key=lambda item: (
                item["macro_f1_mean"],
                item["accuracy_mean"],
                -AUTO_MODEL_TYPE_CANDIDATES.index(item["model_type"]),
            ),
            reverse=True,
        )
        selected_result = benchmark_results[0]
        benchmark_summary = _format_benchmark_summary(
            benchmark_results,
            selected_result["model_type"],
        )
        benchmark_confusion_matrices = _format_benchmark_confusion_matrices(benchmark_results)
    else:
        X_train, X_test, y_train, y_test = _safe_split(X, y, groups=groups)
        selected_result = _evaluate_split(
            mode,
            safe_requested_model_type,
            X_train,
            X_test,
            y_train,
            y_test,
            labels=all_labels,
        )

    safe_model_type = selected_result["model_type"]
    display_model_type = selected_result["model_type_display"]
    report = selected_result["report"]
    confusion_note = (
        "summed across repeated holdout splits"
        if safe_requested_model_type == "auto_best"
        else "holdout test split"
    )
    confusion_matrix_text = _format_confusion_matrix(
        selected_result["confusion_labels"],
        selected_result["confusion_matrix"],
        title=f"{display_model_type} confusion matrix",
        note=confusion_note,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classifier, _safe_model_type_unused, _display_model_type_unused = _build_classifier(
        mode,
        safe_model_type,
        train_size=len(X_scaled),
    )
    classifier.fit(X_scaled, y)

    return {
        "mode": mode,
        "classifier": classifier,
        "scaler": scaler,
        "labels": sorted(np.unique(y).tolist()),
        "dataset_dir": str(dataset_source),
        "sample_count": int(len(X)),
        "report": report,
        "confusion_labels": list(selected_result["confusion_labels"]),
        "confusion_matrix": selected_result["confusion_matrix"],
        "confusion_matrix_text": confusion_matrix_text,
        "model_type": safe_model_type,
        "model_type_display": display_model_type,
        "requested_model_type": safe_requested_model_type,
        "requested_model_type_display": model_type_display_name(safe_requested_model_type),
        "benchmark_results": benchmark_results,
        "benchmark_summary": benchmark_summary,
        "benchmark_confusion_matrices": benchmark_confusion_matrices,
        "preprocessing_version": CURRENT_PREPROCESSING_VERSION,
        "feature_names": tuple(feature_names),
        "config": dict(config),
    }


def _single_training_bundle(training_source, dataset_source, selected_channels, model_type, calibration=None):
    X = []
    y = []
    groups = []
    cal_k = float((calibration or {}).get("event_threshold_k", 3.0)) if isinstance(calibration, dict) else 3.0

    if len(selected_channels) == 1:
        ch = selected_channels[0]
        for csv_file in sorted(training_source):
            payload = _load_channel_capture(csv_file)
            if payload is None:
                continue

            sample_calibration = _resolve_sample_calibration(calibration, csv_file)
            ch_cal = (sample_calibration or {}).get("channels", {}).get(ch)
            label = payload["label"]
            fs = _estimate_sampling_rate(payload["time_s"])
            prepared_signals, has_event = prepare_model_signals(
                [payload["signal"]],
                channel_calibrations=[ch_cal],
                fs=fs,
                label=label,
                k=cal_k,
                fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                keep_full_window=(label == "rest" or ch_cal is None),
            )
            if prepared_signals is None:
                continue

            sig = prepared_signals[0]
            X.append(extract_single_features(sig, noise_std=_noise_std(ch_cal), feature_names=SINGLE_FEATURE_NAMES))
            y.append(label)
            groups.append(str(csv_file))
    else:
        for record in training_source:
            loaded_group = _load_group_capture(record, selected_channels)
            if loaded_group is None:
                continue

            channel_payloads, label = loaded_group
            sample_calibration = _resolve_sample_calibration(calibration, record)
            cal_channels = (sample_calibration or {}).get("channels", {})
            min_len = min(len(channel_payloads[ch]["signal"]) for ch in selected_channels)
            if min_len < 10:
                continue

            aligned_signals = [channel_payloads[ch]["signal"][:min_len] for ch in selected_channels]
            time_s = channel_payloads[selected_channels[0]]["time_s"][:min_len]
            fs = _estimate_sampling_rate(time_s)
            channel_calibrations = [cal_channels.get(ch) for ch in selected_channels]
            prepared_signals, has_event = prepare_model_signals(
                aligned_signals,
                channel_calibrations=channel_calibrations,
                fs=fs,
                label=label,
                k=cal_k,
                fallback_threshold=DEFAULT_FALLBACK_EVENT_THRESHOLD,
                keep_full_window=(label == "rest" or not any(channel_calibrations)),
            )
            if prepared_signals is None:
                continue

            features = [
                extract_single_features(
                    signal,
                    noise_std=_noise_std(channel_calibrations[idx]),
                    feature_names=SINGLE_FEATURE_NAMES,
                )
                for idx, signal in enumerate(prepared_signals)
            ]
            X.append(np.concatenate(features).astype(np.float32))
            y.append(label)
            groups.append(str(next(iter(record["files_by_channel"].values()))))

    if not X:
        raise ValueError("No usable single-gesture samples were found for the selected channels.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    return _train_feature_bundle(
        "single",
        X,
        y,
        groups,
        dataset_source,
        model_type,
        SINGLE_FEATURE_NAMES,
        {
            "buffer_seconds": 3.0,
            "analysis_window_seconds": DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS,
            "predict_every_seconds": 0.2,
            "min_points_to_predict": 50,
            "rest_threshold": DEFAULT_FALLBACK_EVENT_THRESHOLD,
            "confidence_threshold": 0.50,
            "vote_window": 5,
            "event_quiet_seconds": DEFAULT_EVENT_QUIET_SECONDS,
            "event_completion_max_lag_seconds": DEFAULT_EVENT_COMPLETION_MAX_LAG_SECONDS,
            "event_tail_active_fraction": DEFAULT_EVENT_TAIL_ACTIVE_FRACTION,
        },
    )


def train_named_model(mode, model_name, selected_users=None, selected_labels=None, channel=DEFAULT_CHANNEL, selected_channels=None, model_type="random_forest", calibration=None):
    """Train and save a named single-gesture model bundle.

    The GUI calls this after the user chooses users, labels, channels, and model
    type. The function loads matching captures, extracts feature rows, evaluates
    or auto-selects the classifier type, refits the final scaler/classifier on
    all selected data, and writes a ``.joblib`` bundle under ``database/models``.
    """
    if mode != "single":
        raise ValueError(f"Only 'single' mode is supported. Got: {mode}")
    selected_channel_names, channel_key = normalize_model_channels(selected_channels=selected_channels, channel=channel)
    safe_model_type = sanitize_model_type(model_type)
    selected_user_names = None
    selected_label_names = None
    if selected_users is not None:
        selected_user_names = [
            sanitize_user_name(user_name)
            for user_name in selected_users
        ]
        if not selected_user_names:
            raise ValueError("Select at least one user before training.")
    if selected_labels is not None:
        selected_label_names = [
            sanitize_label_name(label)
            for label in selected_labels
        ]
        if not selected_label_names:
            raise ValueError("Select at least one label before training.")

    if len(selected_channel_names) == 1:
        training_source = list_data_files(
            mode,
            selected_users=selected_user_names,
            selected_labels=selected_label_names,
            selected_channels=selected_channel_names,
        )
    else:
        training_source = list_capture_groups(
            mode=mode,
            selected_users=selected_user_names,
            selected_labels=selected_label_names,
            selected_channels=selected_channel_names,
        )

    if not training_source:
        raise ValueError("No matching csv files were found for the selected users, labels, and channels.")

    if selected_user_names:
        dataset_source = ",".join(sorted(set(selected_user_names)))
    else:
        dataset_source = "all_users"

    # Auto-load calibration per user when available.
    if calibration is None and selected_user_names:
        calibration_map = {
            user_name: user_calibration
            for user_name in selected_user_names
            for user_calibration in [load_calibration(user_name)]
            if user_calibration is not None
        }
        if len(calibration_map) == 1:
            calibration = next(iter(calibration_map.values()))
        elif calibration_map:
            calibration = calibration_map

    bundle = _single_training_bundle(
        training_source,
        dataset_source,
        selected_channel_names,
        safe_model_type,
        calibration=calibration,
    )

    safe_model_name = sanitize_model_name(model_name)
    bundle["model_name"] = safe_model_name
    bundle["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    bundle["selected_users"] = sorted(set(selected_user_names or []))
    bundle["selected_labels"] = sorted(set(selected_label_names or []))
    bundle["selected_channels"] = list(selected_channel_names)
    bundle["channel"] = channel_key

    model_dir = model_dir_for_mode(mode, selected_channels=selected_channel_names)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for_name(mode, safe_model_name, selected_channels=selected_channel_names)
    _save_training_confusion_heatmaps(bundle, model_path)
    joblib.dump(bundle, model_path)

    return {
        "bundle": bundle,
        "model_name": safe_model_name,
        "model_path": model_path,
    }


def continue_training_from_base(mode, base_model_name, new_model_name,
                                additional_users=None, selected_channels=None,
                                channel=DEFAULT_CHANNEL, calibration=None):
    """Train a new feature model from a saved base model plus additional users.

    The feature-based scikit-learn models used here are retrained from the
    underlying captured CSVs; this keeps SVM/KNN/RandomForest/LogReg behavior
    consistent even though they do not all support true incremental updates.
    """
    safe_base_name = sanitize_model_name(base_model_name)
    safe_new_name = sanitize_model_name(new_model_name)
    if safe_new_name == safe_base_name:
        raise ValueError("Enter a new model name. Continuing training should not overwrite the base model.")

    base_bundle = load_named_model(
        mode,
        safe_base_name,
        channel=channel,
        selected_channels=selected_channels,
    )
    base_users = [
        sanitize_user_name(user_name)
        for user_name in (base_bundle.get("selected_users") or [])
    ]
    if not base_users:
        raise ValueError("The base model does not store its training users. Train a fresh model instead.")

    added_users = [
        sanitize_user_name(user_name)
        for user_name in (additional_users or [])
    ]
    added_users = sorted({user_name for user_name in added_users if user_name})
    new_users = [user_name for user_name in added_users if user_name not in set(base_users)]
    if not new_users:
        raise ValueError("Select at least one user that was not already used by the base model.")

    base_channels = list(base_bundle.get("selected_channels") or [])
    if not base_channels:
        base_channels, _channel_key = normalize_model_channels(
            selected_channels=selected_channels,
            channel=base_bundle.get("channel", channel),
        )

    base_labels = list(base_bundle.get("labels") or [])
    if not base_labels:
        raise ValueError("The base model does not store its label list. Train a fresh model instead.")

    if len(base_channels) == 1:
        additional_source = list_data_files(
            mode,
            selected_users=new_users,
            selected_labels=base_labels,
            selected_channels=base_channels,
        )
    else:
        additional_source = list_capture_groups(
            mode=mode,
            selected_users=new_users,
            selected_labels=base_labels,
            selected_channels=base_channels,
        )
    if not additional_source:
        raise ValueError(
            "No matching csv files were found for the additional users with the base model's labels and channels."
        )

    combined_users = sorted(set(base_users + new_users))
    result = train_named_model(
        mode,
        safe_new_name,
        selected_users=combined_users,
        selected_labels=base_labels,
        selected_channels=base_channels,
        model_type=base_bundle.get("model_type") or base_bundle.get("requested_model_type") or "random_forest",
        calibration=calibration,
    )

    bundle = result["bundle"]
    bundle["continued_from_model"] = safe_base_name
    bundle["continued_from_created_at"] = base_bundle.get("created_at")
    bundle["continued_from_users"] = sorted(set(base_users))
    bundle["continued_with_users"] = new_users
    bundle["continued_training_strategy"] = "retrained_from_base_and_additional_user_captures"
    joblib.dump(bundle, result["model_path"])
    return result


def _majority_vote(history, default_value):
    if not history:
        return default_value
    counter = Counter(history)
    return counter.most_common(1)[0][0]


def _tail_window_slice(time_signal, window_seconds):
    if window_seconds is None or window_seconds <= 0 or len(time_signal) == 0:
        return slice(0, len(time_signal))

    start_time = float(time_signal[-1] - window_seconds)
    start_idx = int(np.searchsorted(time_signal, start_time, side="left"))
    return slice(start_idx, len(time_signal))


class LivePredictor:
    """Runtime predictor used by the Testing view.

    ``LivePredictor`` consumes rolling per-channel buffers from the GUI, detects
    the latest active event slice, extracts the same feature vector used during
    training, applies the saved scaler/classifier, and smooths the displayed
    label with confidence gating plus majority voting.
    """

    def __init__(self, bundle, calibration=None, display_threshold=0.70):
        self.bundle = bundle
        self.mode = bundle["mode"]
        self.classifier = bundle["classifier"]
        self.scaler = bundle.get("scaler")
        self.config = bundle["config"]
        self.is_conv1d = isinstance(self.classifier, Conv1DClassifier) or bundle.get("model_type") == "conv1d"
        if self.is_conv1d:
            raise ValueError(
                "This saved model uses retired Conv1D support. "
                "Retrain it with Random Forest, MLP, KNN, SVM, or Logistic Regression."
            )
        if self.scaler is None:
            raise ValueError("This saved model is missing its feature scaler. Retrain the model.")
        self.pred_history = deque(maxlen=self.config.get("vote_window", 5))
        self.raw_prediction = "N/A"
        self.display_prediction = "idle"
        self.confidence = 0.0
        self.is_idle = True
        self.last_event_end_time = None
        self.tail_active_fraction = None
        self.low_conf_streak = 0

        # Calibration-based event detection
        self.calibration = calibration
        self.event_k = float((calibration or {}).get("event_threshold_k", 3.0))
        self.display_threshold = display_threshold

    def reset(self):
        self.pred_history.clear()
        self.raw_prediction = "N/A"
        self.display_prediction = "idle"
        self.confidence = 0.0
        self.is_idle = True
        self.last_event_end_time = None
        self.tail_active_fraction = None
        self.low_conf_streak = 0

    def set_calibration(self, calibration):
        self.calibration = calibration
        self.event_k = float((calibration or {}).get("event_threshold_k", 3.0))

    def set_display_threshold(self, threshold):
        self.display_threshold = float(threshold)

    def input_channels(self):
        channels = self.bundle.get("selected_channels")
        if channels:
            return normalize_channel_selection(selected_channels=channels)
        return parse_channel_group(self.bundle.get("channel")) or [DEFAULT_CHANNEL]

    def _channel_calibrations(self):
        cal_channels = (self.calibration or {}).get("channels", {})
        return [cal_channels.get(ch) for ch in self.input_channels()]

    def _feature_names(self):
        return feature_names_for_bundle(self.bundle)

    def _set_idle(self):
        self.is_idle = True
        self.pred_history.clear()
        self.raw_prediction = "idle"
        self.display_prediction = "idle"
        self.confidence = 0.0
        self.low_conf_streak = 0
        return self.snapshot()

    def _collect_live_window(self, buffers_by_channel, analysis_window_seconds):
        windowed_times = []
        windowed_signals = []

        for channel in self.input_channels():
            channel_buffer = buffers_by_channel.get(channel)
            if channel_buffer is None:
                return None, None

            # IMPORTANT: time must be float64 here. The GUI stores Python
            # wallclock timestamps (~1.7e9 in 2026), and float32 silently
            # collapses adjacent samples to the same value, which breaks
            # _estimate_sampling_rate, lag_seconds, and the dedupe margin
            # (all timestamps round to the same number => dedupe blocks
            # every poll after the first classification, and the predictor
            # ends up stuck on whatever state was last committed).
            current_time = np.asarray(channel_buffer["time"], dtype=np.float64)
            current_signal = np.asarray(channel_buffer["voltage"], dtype=np.float32)
            if len(current_time) < 2 or len(current_signal) < 5:
                return None, None

            window_slice = _tail_window_slice(current_time, analysis_window_seconds)
            current_time = current_time[window_slice]
            current_signal = current_signal[window_slice]
            if len(current_time) < 2 or len(current_signal) < 5:
                return None, None

            windowed_times.append(current_time)
            windowed_signals.append(current_signal)

        min_len = min(len(signal) for signal in windowed_signals)
        if min_len < 5:
            return None, None

        aligned_signals = [signal[-min_len:] for signal in windowed_signals]
        aligned_time = min(windowed_times, key=len)[-min_len:]
        return np.asarray(aligned_time, dtype=np.float64), aligned_signals

    def _run_classifier(self, prepared_signals, noise_stds, feature_extractor):
        feature_names = self._feature_names()
        features = np.concatenate(
            [
                feature_extractor(
                    signal,
                    noise_std=noise_stds[idx],
                    feature_names=feature_names,
                )
                for idx, signal in enumerate(prepared_signals)
            ]
        ).astype(np.float32).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = 0.0
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))
        return prediction, confidence

    def _commit_prediction(self, prediction, confidence):
        self.is_idle = False
        self.raw_prediction = prediction
        self.confidence = confidence

        if confidence >= self.config.get("confidence_threshold", 0.5):
            self.pred_history.append(prediction)

        voted = _majority_vote(self.pred_history, prediction)

        if confidence >= self.display_threshold:
            # Confident enough — show the voted label immediately.
            self.low_conf_streak = 0
            self.display_prediction = voted
        else:
            # Low-confidence prediction. The first 2-3 polls of a new gesture
            # are often low-confidence because the event slice is still short
            # (the classifier only sees the opening fragment of the burst).
            # Show 'capturing' during that warm-up period, and only commit to
            # 'unknown' after several consecutive low-confidence polls — by
            # then the classifier has seen enough signal and is genuinely
            # uncertain, not still ramping up.
            self.low_conf_streak += 1
            low_conf_grace = int(
                self.config.get("unknown_after_low_conf_polls", 3)
            )
            if self.low_conf_streak >= low_conf_grace:
                self.display_prediction = "unknown"
            else:
                self.display_prediction = "capturing"
        return self.snapshot()

    def predict(self, buffers_by_channel):
        return self._predict_single(buffers_by_channel)

    def _predict_single(self, buffers_by_channel):
        analysis_window_seconds = max(
            self.config.get("analysis_window_seconds", DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS),
            self.config.get("buffer_seconds", DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS),
        )
        aligned_time, aligned_signals = self._collect_live_window(
            buffers_by_channel,
            analysis_window_seconds=analysis_window_seconds,
        )
        if aligned_time is None or aligned_signals is None:
            return self.snapshot()

        if min(len(signal) for signal in aligned_signals) < self.config.get("min_points_to_predict", 50):
            return self.snapshot()

        fs = _estimate_sampling_rate(aligned_time)
        channel_calibrations = self._channel_calibrations()
        centered_signals, combined_mask, event_slice = analyze_model_window(
            aligned_signals,
            channel_calibrations=channel_calibrations,
            fs=fs,
            k=self.event_k,
            fallback_threshold=self.config.get("rest_threshold", DEFAULT_FALLBACK_EVENT_THRESHOLD),
            slice_strategy="latest",
        )

        # Track tail activity fraction for the GUI status line (diagnostic
        # only — not used to gate prediction anymore).
        quiet_seconds = float(self.config.get("event_quiet_seconds", DEFAULT_EVENT_QUIET_SECONDS))
        quiet_mask = aligned_time >= float(aligned_time[-1] - quiet_seconds)
        tail_samples = combined_mask[quiet_mask]
        if len(tail_samples) > 0:
            self.tail_active_fraction = float(np.sum(tail_samples)) / float(len(tail_samples))
        else:
            self.tail_active_fraction = 0.0

        if event_slice is None:
            # No full event yet. If activity has *just* started (the very last
            # sample is above threshold) show a brief 'capturing' indicator so
            # the user knows the system noticed them; otherwise we are idle.
            if len(combined_mask) > 0 and bool(combined_mask[-1]):
                self.is_idle = False
                self.display_prediction = "capturing"
                self.raw_prediction = "capturing"
                self.confidence = 0.0
                return self.snapshot()
            return self._set_idle()

        # How much time elapsed since the slice ended?
        event_end_time = float(aligned_time[event_slice.stop - 1])
        lag_seconds = float(aligned_time[-1] - event_end_time)
        max_completion_lag = float(
            self.config.get(
                "event_completion_max_lag_seconds",
                DEFAULT_EVENT_COMPLETION_MAX_LAG_SECONDS,
            )
        )
        # If the slice ended so long ago that we've clearly moved past this
        # gesture, fall back to idle so a fresh prediction can start next.
        if lag_seconds > max_completion_lag:
            return self._set_idle()

        # Dedupe: if the event end hasn't moved forward meaningfully since the
        # last prediction, return the cached snapshot instead of reclassifying
        # the exact same slice. Using a short margin (50ms) means a growing
        # slice (ongoing gesture) re-predicts on every poll, voting window
        # smooths it, but a fully settled slice stops spamming the classifier.
        dedupe_margin = 0.05
        if self.last_event_end_time is not None and event_end_time <= (self.last_event_end_time + dedupe_margin):
            return self.snapshot()

        prepared_signals = [signal[event_slice] for signal in centered_signals]

        prediction, confidence = self._run_classifier(
            prepared_signals,
            noise_stds=[_noise_std(calibration) for calibration in channel_calibrations],
            feature_extractor=extract_single_features,
        )
        snapshot = self._commit_prediction(prediction, confidence)
        self.last_event_end_time = event_end_time
        return snapshot

    def debug_dump(self, buffers_by_channel):
        """Run the full prediction pipeline but return a report of every
        intermediate value instead of updating state. Never modifies
        last_event_end_time or display_prediction. Intended for the GUI's
        'Debug Predictor' button — the caller formats and prints this."""
        report = {}
        report["input_channels"] = list(self.input_channels())
        report["feature_names_count"] = len(self._feature_names())
        report["event_k"] = float(self.event_k)

        analysis_window_seconds = max(
            float(self.config.get("analysis_window_seconds", DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS)),
            float(self.config.get("buffer_seconds", DEFAULT_SINGLE_ANALYSIS_WINDOW_SECONDS)),
        )
        report["analysis_window_seconds"] = analysis_window_seconds

        aligned_time, aligned_signals = self._collect_live_window(
            buffers_by_channel,
            analysis_window_seconds=analysis_window_seconds,
        )
        if aligned_time is None or aligned_signals is None:
            report["error"] = "collect_live_window returned None (buffer too short or channels missing)"
            return report

        n_samples = int(min(len(s) for s in aligned_signals))
        min_points = int(self.config.get("min_points_to_predict", 50))
        report["aligned_sample_count"] = n_samples
        report["aligned_duration_seconds"] = float(aligned_time[-1] - aligned_time[0])
        report["min_points_to_predict"] = min_points
        if n_samples < min_points:
            report["error"] = f"aligned_sample_count {n_samples} < min_points_to_predict {min_points}"
            return report

        fs = _estimate_sampling_rate(aligned_time)
        report["fs_estimated_hz"] = float(fs)

        channel_calibrations = self._channel_calibrations()
        fallback_threshold = float(self.config.get("rest_threshold", DEFAULT_FALLBACK_EVENT_THRESHOLD))
        report["fallback_threshold"] = fallback_threshold

        per_channel = {}
        channel_names = list(self.input_channels())
        for idx, (ch, sig, cal) in enumerate(zip(channel_names, aligned_signals, channel_calibrations)):
            sig_arr = np.asarray(sig, dtype=np.float32)
            cal_sane = _channel_calibration_is_sane(cal)
            baseline = _baseline_mean(cal, sig_arr)
            noise_std = _noise_std(cal)
            threshold = _event_threshold(
                cal, k=self.event_k, fallback_threshold=fallback_threshold
            )
            centered = sig_arr - baseline
            envelope = _smooth_envelope(centered)
            mask = envelope > threshold

            runs = []
            cur_start = None
            for i, v in enumerate(mask):
                if v and cur_start is None:
                    cur_start = i
                elif not v and cur_start is not None:
                    runs.append((cur_start, i - cur_start))
                    cur_start = None
            if cur_start is not None:
                runs.append((cur_start, len(mask) - cur_start))

            per_channel[ch] = {
                "calibration_loaded": cal is not None,
                "calibration_sane": cal_sane,
                "cal_voltage_mean": float(cal.get("voltage_mean", 0.0)) if cal is not None else None,
                "cal_voltage_std": float(cal.get("voltage_std", 0.0)) if cal is not None else None,
                "baseline_used": float(baseline),
                "noise_std_used": float(noise_std) if noise_std is not None else None,
                "threshold": float(threshold),
                "raw_signal_min": float(sig_arr.min()),
                "raw_signal_max": float(sig_arr.max()),
                "raw_signal_mean": float(sig_arr.mean()),
                "centered_min": float(centered.min()),
                "centered_max": float(centered.max()),
                "centered_absmax": float(np.max(np.abs(centered))),
                "envelope_min": float(envelope.min()),
                "envelope_max": float(envelope.max()),
                "envelope_mean": float(envelope.mean()),
                "mask_true_count": int(np.sum(mask)),
                "mask_true_fraction": float(np.mean(mask)),
                "mask_last_sample_active": bool(mask[-1]) if len(mask) > 0 else False,
                "num_runs": len(runs),
                "longest_run": max((r[1] for r in runs), default=0),
                "first_5_runs_start_len": runs[:5],
            }

        report["channels"] = per_channel

        centered_signals, combined_mask, event_slice = analyze_model_window(
            aligned_signals,
            channel_calibrations=channel_calibrations,
            fs=fs,
            k=self.event_k,
            fallback_threshold=fallback_threshold,
            slice_strategy="latest",
        )
        min_samples, pad_samples = _event_sample_counts(fs)
        report["min_samples_required"] = int(min_samples)
        report["pad_samples"] = int(pad_samples)
        report["combined_mask_true_count"] = int(np.sum(combined_mask))
        report["combined_mask_true_fraction"] = float(np.mean(combined_mask))
        report["combined_mask_last_sample_active"] = bool(combined_mask[-1]) if len(combined_mask) > 0 else False

        quiet_seconds = float(self.config.get("event_quiet_seconds", DEFAULT_EVENT_QUIET_SECONDS))
        quiet_mask = aligned_time >= float(aligned_time[-1] - quiet_seconds)
        tail_samples = combined_mask[quiet_mask]
        if len(tail_samples) > 0:
            report["tail_active_fraction"] = float(np.sum(tail_samples)) / float(len(tail_samples))
        else:
            report["tail_active_fraction"] = 0.0

        if event_slice is None:
            report["event_slice"] = None
            report["would_display"] = (
                "capturing"
                if len(combined_mask) > 0 and bool(combined_mask[-1])
                else "idle"
            )
            return report

        report["event_slice_start"] = int(event_slice.start)
        report["event_slice_stop"] = int(event_slice.stop)
        report["event_slice_length"] = int(event_slice.stop - event_slice.start)
        report["event_slice_duration_seconds"] = float(
            aligned_time[event_slice.stop - 1] - aligned_time[event_slice.start]
        )

        event_end_time = float(aligned_time[event_slice.stop - 1])
        lag_seconds = float(aligned_time[-1] - event_end_time)
        max_completion_lag = float(
            self.config.get(
                "event_completion_max_lag_seconds",
                DEFAULT_EVENT_COMPLETION_MAX_LAG_SECONDS,
            )
        )
        report["event_end_time"] = event_end_time
        report["lag_seconds"] = lag_seconds
        report["max_completion_lag"] = max_completion_lag

        if lag_seconds > max_completion_lag:
            report["would_display"] = f"idle (stale — lag {lag_seconds:.3f}s > {max_completion_lag:.3f}s)"
            return report

        try:
            prepared_signals = [signal[event_slice] for signal in centered_signals]
            prediction, confidence = self._run_classifier(
                prepared_signals,
                noise_stds=[_noise_std(c) for c in channel_calibrations],
                feature_extractor=extract_single_features,
            )
            report["classifier_prediction"] = str(prediction)
            report["classifier_confidence"] = float(confidence)
            report["confidence_threshold"] = float(self.config.get("confidence_threshold", 0.5))
            report["display_threshold"] = float(self.display_threshold)
            if confidence < self.display_threshold:
                report["would_display"] = f"unknown (conf {confidence:.2f} < display_threshold {self.display_threshold:.2f})"
            else:
                report["would_display"] = f"{prediction} (conf {confidence:.2f})"
        except Exception as exc:
            report["classifier_error"] = str(exc)
            report["would_display"] = "classifier exception"

        return report

    def snapshot(self):
        return {
            "display_prediction": self.display_prediction,
            "raw_prediction": self.raw_prediction,
            "confidence": self.confidence,
            "is_idle": self.is_idle,
            "tail_active_fraction": self.tail_active_fraction,
        }

    @property
    def predict_every_seconds(self):
        return self.config.get("predict_every_seconds", 0.1)


def fine_tune_model(base_bundle_path, mode, user_name, model_name,
                    selected_channels=None, channel=DEFAULT_CHANNEL,
                    epochs=20, lr=5e-4):
    raise ValueError(
        "Conv1D fine-tuning has been removed. "
        "Train a personal feature-based model directly from the selected user's data instead."
    )
