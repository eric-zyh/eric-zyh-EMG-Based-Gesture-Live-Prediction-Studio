"""Filesystem and metadata helpers for EMGesture.

This module defines the app's on-disk data contract: users, labels, channel
folders, capture CSV files, calibration JSON files, and model discovery paths.
The GUI and model code should go through these helpers instead of constructing
database paths or CSV schemas directly.
"""

import csv as csv_mod
import json
import math
import re
import shutil
import time
from pathlib import Path


CHANNEL_NAMES = ("a0", "a1")
DEFAULT_CHANNEL = "a0"
DB_ROOT = Path(__file__).parent / "database"
CALIBRATION_DIR = DB_ROOT / "calibrations"

MODE_DIR_MAP = {
    "single": "emg_dataset",
}

DEFAULT_LABELS = {
    "single": ["rest", "fist", "open", "one"],
}


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def sanitize_user_name(raw_name):
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw_name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or f"user_{int(time.time())}"


def sanitize_label_name(raw_name):
    cleaned = re.sub(r"[^a-z0-9_]", "_", raw_name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Label name cannot be empty.")
    return cleaned


def sanitize_channel_name(raw_name):
    cleaned = raw_name.strip().lower()
    if cleaned not in CHANNEL_NAMES:
        raise ValueError(
            f"Unknown channel: {raw_name}. Valid channels: {', '.join(CHANNEL_NAMES)}"
        )
    return cleaned


def normalize_channel_selection(selected_channels=None, channel=DEFAULT_CHANNEL):
    """Normalize GUI/channel inputs to a non-empty list of supported channels."""
    if selected_channels:
        channels = []
        for ch in selected_channels:
            try:
                channels.append(sanitize_channel_name(ch))
            except ValueError:
                continue
        return channels or [DEFAULT_CHANNEL]
    try:
        return [sanitize_channel_name(channel)]
    except ValueError:
        return [DEFAULT_CHANNEL]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mode_dir_name(mode):
    if mode not in MODE_DIR_MAP:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {', '.join(MODE_DIR_MAP)}")
    return MODE_DIR_MAP[mode]


def _labels_file(mode):
    return DB_ROOT / "labels" / f"{mode}.json"


def _label_catalog_file():
    return DB_ROOT / "label_catalog.json"


def _merge_labels(preferred, fallback):
    labels = []
    for label in list(preferred or []) + list(fallback or []):
        if label and label not in labels:
            labels.append(label)
    return labels


def _load_labels(mode):
    path = _labels_file(mode)
    if path.exists():
        try:
            return _merge_labels(json.loads(path.read_text()), DEFAULT_LABELS.get(mode, []))
        except (json.JSONDecodeError, OSError):
            pass

    catalog_path = _label_catalog_file()
    if catalog_path.exists():
        try:
            catalog = json.loads(catalog_path.read_text())
            return _merge_labels(catalog.get(mode, []), DEFAULT_LABELS.get(mode, []))
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    return list(DEFAULT_LABELS.get(mode, []))


def _save_labels(mode, labels):
    path = _labels_file(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2))


def _user_mode_dir(user_name, mode):
    return DB_ROOT / "users" / sanitize_user_name(user_name) / _mode_dir_name(mode)


def _user_channel_dir(user_name, mode, channel):
    return _user_mode_dir(user_name, mode) / sanitize_channel_name(channel)


def _read_csv_label(csv_file):
    try:
        with csv_file.open("r", newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                return row.get("label", "")
    except (OSError, KeyError, StopIteration):
        pass
    return ""


# ---------------------------------------------------------------------------
# Label management
# ---------------------------------------------------------------------------

def list_labels_for_mode(mode):
    return _load_labels(mode)


def add_label_to_mode(mode, raw_label):
    label = sanitize_label_name(raw_label)
    labels = _load_labels(mode)
    if label not in labels:
        labels.append(label)
        _save_labels(mode, labels)
    return label


def remove_label_from_mode(mode, raw_label):
    label = sanitize_label_name(raw_label)
    labels = _load_labels(mode)
    if label not in labels:
        raise ValueError(f"Label '{label}' does not exist in {mode} mode.")
    if label == "rest":
        raise ValueError("Cannot remove the 'rest' label.")
    labels.remove(label)
    _save_labels(mode, labels)
    return label


# ---------------------------------------------------------------------------
# Channel label management (muscle names)
# ---------------------------------------------------------------------------

def _channel_labels_file():
    return DB_ROOT / "channel_labels.json"


def _load_channel_labels():
    path = _channel_labels_file()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_channel_labels(labels):
    path = _channel_labels_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2))


def get_channel_labels():
    """Return dict mapping channel id -> muscle name, e.g. {"a0": "forearm", "a1": "finger"}."""
    return _load_channel_labels()


def set_channel_label(channel, muscle_name):
    """Assign a muscle name to a channel. Returns the sanitized name."""
    ch = sanitize_channel_name(channel)
    cleaned = re.sub(r"[^a-z0-9_ ]", "", muscle_name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Muscle name cannot be empty.")
    labels = _load_channel_labels()
    labels[ch] = cleaned
    _save_channel_labels(labels)
    return cleaned


def remove_channel_label(channel):
    """Remove muscle name from a channel, reverting to raw channel id."""
    ch = sanitize_channel_name(channel)
    labels = _load_channel_labels()
    if ch in labels:
        del labels[ch]
        _save_channel_labels(labels)


def channel_display_name(channel):
    """Return muscle name if set, otherwise the raw channel id (e.g. 'A0')."""
    labels = _load_channel_labels()
    return labels.get(channel, channel.upper())


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def list_available_channels(mode, selected_users=None):
    channels = set()
    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return sorted(channels)

    target_users = (
        {sanitize_user_name(u) for u in selected_users} if selected_users else None
    )

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        if target_users and user_dir.name not in target_users:
            continue
        mode_dir = user_dir / _mode_dir_name(mode)
        if not mode_dir.exists():
            continue
        for ch_dir in mode_dir.iterdir():
            if ch_dir.is_dir() and ch_dir.name in CHANNEL_NAMES:
                if any(ch_dir.glob("*.csv")):
                    channels.add(ch_dir.name)

    return sorted(channels)


def list_available_users(mode, selected_channels=None):
    users = []
    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return users

    target_channels = None
    if selected_channels:
        target_channels = {
            sanitize_channel_name(ch)
            for ch in selected_channels
            if ch in CHANNEL_NAMES
        }

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        mode_dir = user_dir / _mode_dir_name(mode)
        if not mode_dir.exists():
            continue

        for ch_dir in mode_dir.iterdir():
            if not ch_dir.is_dir() or ch_dir.name not in CHANNEL_NAMES:
                continue
            if target_channels and ch_dir.name not in target_channels:
                continue
            if any(ch_dir.glob("*.csv")):
                users.append(user_dir.name)
                break

    return users


def list_available_labels(mode, selected_users=None, selected_channels=None):
    labels = set()
    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return sorted(labels)

    target_users = (
        {sanitize_user_name(u) for u in selected_users} if selected_users else None
    )
    target_channels = None
    if selected_channels:
        target_channels = {
            sanitize_channel_name(ch)
            for ch in selected_channels
            if ch in CHANNEL_NAMES
        }

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        if target_users and user_dir.name not in target_users:
            continue
        mode_dir = user_dir / _mode_dir_name(mode)
        if not mode_dir.exists():
            continue

        for ch_dir in mode_dir.iterdir():
            if not ch_dir.is_dir() or ch_dir.name not in CHANNEL_NAMES:
                continue
            if target_channels and ch_dir.name not in target_channels:
                continue
            for csv_file in ch_dir.glob("*.csv"):
                label = _read_csv_label(csv_file)
                if label:
                    labels.add(label)

    return sorted(labels)


def list_data_files(mode, selected_users=None, selected_labels=None, selected_channels=None):
    files = []
    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return files

    target_users = (
        {sanitize_user_name(u) for u in selected_users} if selected_users else None
    )
    target_labels = (
        {sanitize_label_name(l) for l in selected_labels} if selected_labels else None
    )
    channels = normalize_channel_selection(selected_channels=selected_channels)

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        if target_users and user_dir.name not in target_users:
            continue
        mode_dir = user_dir / _mode_dir_name(mode)
        if not mode_dir.exists():
            continue

        for channel in channels:
            ch_dir = mode_dir / channel
            if not ch_dir.exists():
                continue
            for csv_file in sorted(ch_dir.glob("*.csv")):
                if target_labels:
                    label = _read_csv_label(csv_file)
                    if label in target_labels:
                        files.append(csv_file)
                else:
                    files.append(csv_file)

    return files


def list_capture_groups(mode, selected_users=None, selected_labels=None, selected_channels=None):
    """Return multi-channel capture records with matching filenames.

    A capture group represents the same gesture repetition recorded on multiple
    channels. Training uses these groups to concatenate synchronized per-channel
    features while keeping one label and one group id for the capture.
    """
    channels = normalize_channel_selection(selected_channels=selected_channels)
    if len(channels) < 2:
        return []

    groups = {}
    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return []

    target_users = (
        {sanitize_user_name(u) for u in selected_users} if selected_users else None
    )
    target_labels = (
        {sanitize_label_name(l) for l in selected_labels} if selected_labels else None
    )

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        if target_users and user_dir.name not in target_users:
            continue
        mode_dir = user_dir / _mode_dir_name(mode)
        if not mode_dir.exists():
            continue

        for channel in channels:
            ch_dir = mode_dir / channel
            if not ch_dir.exists():
                continue
            for csv_file in sorted(ch_dir.glob("*.csv")):
                filename = csv_file.name
                if filename not in groups:
                    groups[filename] = {"filename": filename, "files_by_channel": {}}
                groups[filename]["files_by_channel"][channel] = csv_file

    channel_set = set(channels)
    result = []
    for filename, record in sorted(groups.items()):
        if set(record["files_by_channel"].keys()) >= channel_set:
            if target_labels:
                first_file = next(iter(record["files_by_channel"].values()))
                label = _read_csv_label(first_file)
                if label in target_labels:
                    result.append(record)
            else:
                result.append(record)

    return result


def list_user_capture_records(user_name):
    safe_user = sanitize_user_name(user_name)
    user_root = DB_ROOT / "users" / safe_user
    if not user_root.exists():
        return []

    records = []
    for mode, mode_dir_name in MODE_DIR_MAP.items():
        mode_dir = user_root / mode_dir_name
        if not mode_dir.exists():
            continue

        file_groups = {}
        for ch_dir in sorted(mode_dir.iterdir()):
            if not ch_dir.is_dir() or ch_dir.name not in CHANNEL_NAMES:
                continue
            channel = ch_dir.name
            for csv_file in sorted(ch_dir.glob("*.csv")):
                filename = csv_file.name
                if filename not in file_groups:
                    file_groups[filename] = {
                        "filename": filename,
                        "mode": mode,
                        "channels": [],
                        "files_by_channel": {},
                        "label": "",
                        "timestamp": "",
                    }
                file_groups[filename]["channels"].append(channel)
                file_groups[filename]["files_by_channel"][channel] = csv_file

        for record in file_groups.values():
            first_file = next(iter(record["files_by_channel"].values()))
            record["label"] = _read_csv_label(first_file) or "unknown"
            stat = first_file.stat()
            record["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
            )
            records.append(record)

    records.sort(key=lambda r: r["timestamp"], reverse=True)
    return records


# ---------------------------------------------------------------------------
# Data writing
# ---------------------------------------------------------------------------

def next_shared_capture_index(user_name, mode, label, channels):
    safe_user = sanitize_user_name(user_name)
    max_index = 0
    for channel in channels:
        ch_dir = _user_channel_dir(safe_user, mode, channel)
        if not ch_dir.exists():
            continue
        for csv_file in ch_dir.glob("*.csv"):
            parts = csv_file.stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    max_index = max(max_index, int(parts[1]))
                except ValueError:
                    pass
    return max_index + 1


def write_capture(rows, user_name, mode, label, channel, filename=None):
    """Write one labeled capture CSV and return its path.

    ``rows`` must match the schema consumed by training:
    ``time_s, adc, voltage, label``.
    """
    safe_user = sanitize_user_name(user_name)
    safe_label = sanitize_label_name(label)
    safe_channel = sanitize_channel_name(channel)

    target_dir = _user_channel_dir(safe_user, mode, safe_channel)
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        index = next_shared_capture_index(safe_user, mode, safe_label, [safe_channel])
        filename = f"{safe_user}_emg_{safe_label}_{index:03d}.csv"

    filepath = target_dir / filename
    with filepath.open("w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["time_s", "adc", "voltage", "label"])
        for row in rows:
            writer.writerow(row)

    return filepath


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------

def clear_user_data(user_name):
    safe_user = sanitize_user_name(user_name)
    user_dir = DB_ROOT / "users" / safe_user

    deleted_files = 0
    if user_dir.exists():
        for csv_file in user_dir.rglob("*.csv"):
            csv_file.unlink()
            deleted_files += 1
        for dirpath in sorted(user_dir.rglob("*"), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                dirpath.rmdir()
        if user_dir.exists() and not any(user_dir.iterdir()):
            user_dir.rmdir()

    return {"deleted_files": deleted_files, "user": safe_user}


# ---------------------------------------------------------------------------
# Calibration data I/O
# ---------------------------------------------------------------------------

def calibration_path(user_name):
    """Return path to a user's calibration JSON file."""
    return CALIBRATION_DIR / f"{sanitize_user_name(user_name)}.json"


def save_calibration(user_name, calibration_data):
    """Save calibration data (noise floor) for a user.

    calibration_data should be a dict like:
    {
        "user": "zeric",
        "timestamp": "2026-04-12T15:30:00",
        "channels": {
            "a0": {"voltage_mean": 2.505, "voltage_std": 0.015, "adc_mean": 512.5, "adc_std": 3.2},
            "a1": {"voltage_mean": 2.485, "voltage_std": 0.013, "adc_mean": 508.3, "adc_std": 2.8},
        },
        "duration_seconds": 3.0,
        "sample_count": 1500,
        "event_threshold_k": 3.0,
    }
    """
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    path = calibration_path(user_name)
    with path.open("w") as f:
        json.dump(calibration_data, f, indent=2)
    return path


def load_calibration(user_name):
    """Load calibration data for a user, or None if not calibrated."""
    path = calibration_path(user_name)
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    return infer_calibration_from_rest_data(user_name)


def infer_calibration_from_rest_data(user_name):
    """Estimate calibration from a user's stored rest captures."""
    safe_user = sanitize_user_name(user_name)
    mode_dir = _user_mode_dir(safe_user, "single")
    if not mode_dir.exists():
        return None

    channels_data = {}
    min_samples = None

    for channel in CHANNEL_NAMES:
        ch_dir = mode_dir / channel
        if not ch_dir.exists():
            continue

        count = 0
        voltage_sum = voltage_sq_sum = 0.0
        adc_sum = adc_sq_sum = 0.0

        for csv_file in sorted(ch_dir.glob("*.csv")):
            try:
                with csv_file.open("r", newline="") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        if row.get("label") != "rest":
                            continue
                        voltage = float(row["voltage"])
                        adc = float(row["adc"])
                        voltage_sum += voltage
                        voltage_sq_sum += voltage * voltage
                        adc_sum += adc
                        adc_sq_sum += adc * adc
                        count += 1
            except (OSError, KeyError, ValueError):
                continue

        if count == 0:
            continue

        voltage_mean = voltage_sum / count
        voltage_var = max(voltage_sq_sum / count - voltage_mean * voltage_mean, 0.0)
        adc_mean = adc_sum / count
        adc_var = max(adc_sq_sum / count - adc_mean * adc_mean, 0.0)
        channels_data[channel] = {
            "voltage_mean": voltage_mean,
            "voltage_std": math.sqrt(voltage_var),
            "adc_mean": adc_mean,
            "adc_std": math.sqrt(adc_var),
        }
        if min_samples is None or count < min_samples:
            min_samples = count

    if not channels_data:
        return None

    return {
        "user": safe_user,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channels": channels_data,
        "sample_count": int(min_samples or 0),
        "event_threshold_k": 3.0,
        "source": "single_rest_inference",
    }


def mash_database():
    mash_root = DB_ROOT / "mash"
    counts = {mode: {ch: 0 for ch in CHANNEL_NAMES} for mode in MODE_DIR_MAP}

    if mash_root.exists():
        shutil.rmtree(mash_root)

    users_root = DB_ROOT / "users"
    if not users_root.exists():
        return counts

    for user_dir in sorted(users_root.iterdir()):
        if not user_dir.is_dir():
            continue
        for mode, mode_dir_name in MODE_DIR_MAP.items():
            mode_dir = user_dir / mode_dir_name
            if not mode_dir.exists():
                continue
            for ch_dir in mode_dir.iterdir():
                if not ch_dir.is_dir() or ch_dir.name not in CHANNEL_NAMES:
                    continue
                channel = ch_dir.name
                target = mash_root / mode_dir_name / channel
                target.mkdir(parents=True, exist_ok=True)
                for csv_file in ch_dir.glob("*.csv"):
                    dest = target / csv_file.name
                    if dest.exists():
                        dest = target / f"{user_dir.name}_{csv_file.name}"
                    shutil.copy2(csv_file, dest)
                    counts[mode][channel] += 1

    return counts
