"""Tkinter application layer for EMGesture.

This module owns the user-facing workflows: serial connection, live plotting,
gesture capture, model training/testing screens, calibration, debug export, and
the user-data browser. Lower-level filesystem/data operations live in
``emg_data_tools`` and signal-processing/model operations live in
``emg_model_tools``.

The GUI is intentionally stateful because Tkinter callbacks mutate the current
capture, prediction, and plotting state over time. Keep hardware/protocol
assumptions documented at the parser boundary instead of scattering them across
the event handlers.
"""

import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import serial
    from serial.tools import list_ports
    SERIAL_IMPORT_ERROR = None
    SERIAL_EXCEPTION = serial.SerialException
except ModuleNotFoundError as exc:
    serial = None
    list_ports = None
    SERIAL_IMPORT_ERROR = exc
    SERIAL_EXCEPTION = RuntimeError

from emg_data_tools import (
    CALIBRATION_DIR,
    CHANNEL_NAMES,
    DEFAULT_CHANNEL,
    add_label_to_mode,
    calibration_path,
    clear_user_data,
    get_channel_labels,
    load_calibration,
    save_calibration,
    set_channel_label,
    remove_channel_label,
    channel_display_name,
    list_available_channels,
    list_available_labels,
    list_available_users,
    list_user_capture_records,
    list_labels_for_mode,
    mash_database,
    next_shared_capture_index,
    remove_label_from_mode,
    sanitize_channel_name,
    sanitize_user_name,
    write_capture,
)
from emg_model_tools import (
    LivePredictor,
    continue_training_from_base,
    delete_named_model,
    export_processed_capture_preview,
    list_saved_models,
    load_named_model,
    model_type_labels,
    prepare_capture_preview,
    train_named_model,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BAUD = 115200
PLOT_WINDOW_SECONDS = 5.0
SERIAL_POLL_MS = 20
PLOT_REFRESH_MS = 80
BANNER_REFRESH_MS = 100
ARDUINO_WARMUP_SECONDS = 2.0
ADC_MAX = 1023.0
VREF = 5.0
REST_THRESHOLD = 0.15
PREDICTION_HOLD_SECONDS = 0.5
PREDICTION_HISTORY_LIMIT = 50
PLOT_COLORS = {"a0": "#1f77b4", "a1": "#ff7f0e"}

GUIDED_SINGLE_REPEATS = 30
GUIDED_SINGLE_SECONDS = 2.0
GUIDED_SINGLE_REST_SECONDS = 3.0


def discover_serial_ports(include_pseudo=True):
    """Return serial-port candidates, including pseudo terminals for testing."""
    ports = []
    seen = set()

    if list_ports is not None:
        for port in list_ports.comports():
            device = getattr(port, "device", None)
            if device and device not in seen:
                ports.append(device)
                seen.add(device)

    if include_pseudo:
        pseudo_patterns = (
            "ttys[0-9][0-9][0-9]",
            "pts/[0-9]*",
        )
        for pattern in pseudo_patterns:
            for path in sorted(Path("/dev").glob(pattern)):
                device = str(path)
                if device not in seen:
                    ports.append(device)
                    seen.add(device)

    return ports


def adc_to_voltage(adc_value):
    """Convert a 10-bit Arduino ADC reading to volts using the configured Vref."""
    return (float(adc_value) / ADC_MAX) * VREF


def parse_line(line):
    """Parse one Arduino serial line into channel samples.

    Supported formats:
      * ``adc0`` or ``adc0 adc1``: raw ADC values; timestamp is host wallclock
        and voltage is computed using ``VREF`` / ``ADC_MAX``.
      * ``t_ms,adc0,voltage0`` or ``t_ms,adc0,voltage0,adc1,voltage1``:
        device-timed samples with voltage already computed by the sender.

    Returns ``({channel: (time_s, adc, voltage)}, format_name)`` or ``None`` for
    blank/malformed lines. Channel names are normalized to ``a0`` / ``a1``.
    """
    stripped = line.strip()
    if not stripped:
        return None
    comma_parts = [p.strip() for p in stripped.split(",")]
    if len(comma_parts) in (3, 5):
        try:
            t_ms = float(comma_parts[0])
            channel_samples = {
                "a0": (t_ms / 1000.0, int(comma_parts[1]), float(comma_parts[2])),
            }
            fmt = "timed A0"
            if len(comma_parts) == 5:
                channel_samples["a1"] = (t_ms / 1000.0, int(comma_parts[3]), float(comma_parts[4]))
                fmt = "timed A0+A1"
        except ValueError:
            return None
        return channel_samples, fmt

    ws_parts = stripped.split()
    if len(ws_parts) in (1, 2):
        try:
            adc0 = int(ws_parts[0])
            t = time.time()
            channel_samples = {"a0": (t, adc0, adc_to_voltage(adc0))}
            fmt = "raw ADC A0"
            if len(ws_parts) == 2:
                adc1 = int(ws_parts[1])
                channel_samples["a1"] = (t, adc1, adc_to_voltage(adc1))
                fmt = "raw ADC A0+A1"
        except ValueError:
            return None
        return channel_samples, fmt
    return None


# ===================================================================
# Main GUI
# ===================================================================
class EMGCollectorGUI:
    """Stateful Tkinter controller for the full EMG workflow.

    The class is organized around screens and periodic callbacks rather than a
    pure MVC split: Tk variables, serial buffers, capture state, and predictor
    state all live here because they are updated by Tk's event loop. Expensive or
    reusable logic should stay in ``emg_data_tools`` or ``emg_model_tools``.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("EMG Data System")
        self.root.geometry("1400x860")

        # --- shared variables ---
        self.user_name_var = tk.StringVar()
        self.port_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")

        # --- serial state ---
        self.ser = None
        self.serial_ready_at = 0.0
        self.detected_channels = {DEFAULT_CHANNEL}
        self.channel_buffers = {
            ch: {"time": deque(), "adc": deque(), "voltage": deque()}
            for ch in CHANNEL_NAMES
        }

        # --- capture state ---
        self.session_user = None
        self.session_mode = None  # "single" | "session"
        self.prepare_state = None
        self.active_capture = None
        self.auto_running = False
        self.auto_queue = []

        # --- collection page variables ---
        self.selected_label_var = tk.StringVar(value="fist")
        self.single_auto_count_var = tk.IntVar(value=5)
        self.single_window_seconds_var = tk.DoubleVar(value=3.0)
        self.prepare_seconds_var = tk.DoubleVar(value=2.0)
        self.new_label_var = tk.StringVar()
        self.plot_unit_var = tk.StringVar(value="adc")

        # --- guided session state ---
        self.guided_single_repeats_var = tk.IntVar(value=GUIDED_SINGLE_REPEATS)
        self.guided_single_seconds_var = tk.DoubleVar(value=GUIDED_SINGLE_SECONDS)
        self.guided_single_rest_seconds_var = tk.DoubleVar(value=GUIDED_SINGLE_REST_SECONDS)
        self.guided_break_enabled_var = tk.BooleanVar(value=True)
        self.guided_session_active = False
        self.guided_session_auto = False
        self.guided_session_waiting = False
        self.guided_session_paused = False
        self.guided_session_tasks = []
        self.guided_session_step_index = 0
        self.guided_break_state = None
        self.guided_session_labels_listbox = None

        # --- training page variables ---
        self.model_name_var = tk.StringVar()
        self.training_model_type_var = tk.StringVar(value=model_type_labels()[0])
        self.home_train_user_listbox = None
        self.home_train_label_listbox = None
        self.home_train_channel_listbox = None
        self.training_mode_var = tk.StringVar(value="single")
        self.training_report_var = tk.StringVar(value="Training results will appear here.")
        self.ft_new_model_name_var = tk.StringVar()
        self._last_finetune_base_name = ""

        # --- testing page variables ---
        self.model_choice_var = tk.StringVar()
        self.model_channel_var = tk.StringVar(value=DEFAULT_CHANNEL)
        self.testing_mode_var = tk.StringVar(value="single")
        self.predictor = None
        self.active_model_name = None
        self.last_predict_wall = 0.0
        self.prediction_var = tk.StringVar(value="Prediction: --")
        self.prediction_display_var = tk.StringVar(value="--")
        self.prediction_hold_until = 0.0
        self.prediction_hold_label = None
        self.last_live_prediction_display = "N/A"
        self.prediction_history_tree = None
        self.model_combo = None

        # --- calibration & confidence ---
        self.calibration_data = None
        self.calibration_status_var = tk.StringVar(value="Not calibrated")
        self.confidence_threshold_var = tk.DoubleVar(value=0.70)
        self.threshold_display_var = tk.StringVar(value="0.70")

        # --- user data page ---
        self.ud_mode_filter_var = tk.StringVar(value="all")
        self.ud_plot_mode_var = tk.StringVar(value="overlay")
        self.ud_selected_record = None
        self.ud_last_preview = None
        self.ud_selected_user = None
        # Debug Mode: when on, every live prediction poll auto-captures a
        # debug dump (rate-limited) into the ring buffer below. Export Debug
        # Report writes the ring buffer to a .txt file.
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.debug_autocapture_wall = 0.0
        self.debug_autocapture_interval = 0.5  # seconds between auto captures
        self.last_debug_reports = deque(maxlen=200)
        # compare state
        self.ud_compare_mode = False
        self.ud_compare_record = None
        self.ud_compare_preview = None
        self.ud_compare_user = None
        self.ud_compare_axis = None
        self.ud_compare_btn = None

        # --- plot ---
        self.plot_axis = None
        self.lines_by_channel = {}
        self.canvas = None
        self.last_rest_state = False

        # --- capture history tree ---
        self.capture_tree = None
        self.capture_record_by_item = {}

        # --- banner ---
        self.banner_var = tk.StringVar(value="")
        self.banner_timer_var = tk.StringVar(value="")
        self.banner_detail_var = tk.StringVar(value="")
        self._last_flash_label = None
        self._flash_after_id = None
        self._plot_dirty = False
        self._last_plot_refresh_at = 0.0
        self._last_banner_refresh_at = 0.0

        # --- log ---
        self.log_widgets = {}  # page_name -> Text widget

        # --- build UI ---
        self._build_pages()
        self._show_page("home")
        self.refresh_ports()
        self._log("Application started.")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(SERIAL_POLL_MS, self.poll_serial)

    # ---------------------------------------------------------------
    # Page management
    # ---------------------------------------------------------------
    def _build_pages(self):
        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.pages = {}
        self._build_home_page()
        self._build_training_menu_page()
        self._build_collection_page()
        self._build_training_page()
        self._build_testing_page()
        self._build_user_data_page()

    def _show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        if name == "home":
            self._refresh_known_users()
            self._refresh_home_models()
            self._refresh_channel_labels()

    def _build_log_panel(self, parent, page_name):
        frame = ttk.LabelFrame(parent, text="Log", padding=4)
        frame.pack(fill="x", side="bottom", pady=(4, 0))
        log_text = tk.Text(frame, height=5, wrap="word", font=("Courier", 10), state="disabled",
                           bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=log_text.yview)
        log_text.configure(yscrollcommand=scroll.set)
        log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.log_widgets[page_name] = log_text
        return frame

    def _log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"
        for widget in self.log_widgets.values():
            widget.configure(state="normal")
            widget.insert("end", line)
            widget.see("end")
            widget.configure(state="disabled")

    # ---------------------------------------------------------------
    # Styled button helper (macOS ignores bg/fg on tk.Button & ttk.Button)
    # ---------------------------------------------------------------
    @staticmethod
    def _make_card_button(parent, text, command,
                          font=("Helvetica", 16, "bold"), pady=18):
        bg, fg, hover_bg = "#e0e0e0", "#1a1a1a", "#c8c8c8"
        frame = tk.Frame(parent, bg=bg, cursor="hand2", bd=1,
                         highlightthickness=1, highlightbackground="#999")
        label = tk.Label(frame, text=text, bg=bg, fg=fg, font=font,
                         padx=24, pady=pady, cursor="hand2")
        label.pack(fill="both", expand=True)
        for w in (frame, label):
            w.bind("<Button-1>", lambda e: command())
            w.bind("<Enter>", lambda e, f=frame, l=label: (f.config(bg=hover_bg), l.config(bg=hover_bg)))
            w.bind("<Leave>", lambda e, f=frame, l=label: (f.config(bg=bg), l.config(bg=bg)))
        return frame

    # ---------------------------------------------------------------
    # HOME PAGE
    # ---------------------------------------------------------------
    def _build_home_page(self):
        page = ttk.Frame(self.container, padding=30)
        self.pages["home"] = page

        ttk.Label(page, text="EMG Data System", font=("Helvetica", 28, "bold")).pack(pady=(10, 20))

        # user + port row
        top = ttk.Frame(page)
        top.pack(fill="x", pady=(0, 20))
        ttk.Label(top, text="User Name:", font=("Helvetica", 13)).pack(side="left")
        self.user_combo = ttk.Combobox(top, textvariable=self.user_name_var, width=20, font=("Helvetica", 13))
        self.user_combo.pack(side="left", padx=(6, 30))
        ttk.Label(top, text="Serial Port:", font=("Helvetica", 13)).pack(side="left")
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, width=24, font=("Helvetica", 13))
        self.port_combo.pack(side="left", padx=(6, 10))
        ttk.Button(top, text="Refresh Ports", command=self.refresh_ports).pack(side="left")

        # 4 navigation buttons (2x2 grid)
        btn_frame = tk.Frame(page)
        btn_frame.pack(fill="x", pady=(0, 20))
        for i in range(2):
            btn_frame.columnconfigure(i, weight=1)

        self._make_card_button(btn_frame, "Training\n(Data Collection)",
                               self._go_training_menu).grid(row=0, column=0, padx=10, pady=8, sticky="nsew")
        self._make_card_button(btn_frame, "Testing\n(Live Prediction)",
                               self._go_testing).grid(row=0, column=1, padx=10, pady=8, sticky="nsew")
        self._make_card_button(btn_frame, "Model Training\n(Build Models)",
                               self._go_model_training).grid(row=1, column=0, padx=10, pady=8, sticky="nsew")
        self._make_card_button(btn_frame, "User Data\n(Browse & Plot)",
                               lambda: self._go_user_data()).grid(row=1, column=1, padx=10, pady=8, sticky="nsew")

        # Bottom section: channel labels | existing users | models
        lists_frame = ttk.Frame(page)
        lists_frame.pack(fill="both", expand=True, pady=(0, 10))
        lists_frame.columnconfigure(0, weight=1)
        lists_frame.columnconfigure(1, weight=1)
        lists_frame.columnconfigure(2, weight=1)
        lists_frame.rowconfigure(0, weight=1)

        # Channel labels (muscle names)
        ch_frame = ttk.LabelFrame(lists_frame, text="Channel Labels", padding=10)
        ch_frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        self.ch_label_listbox = tk.Listbox(ch_frame, font=("Helvetica", 12), height=6,
                                           exportselection=False)
        self.ch_label_listbox.pack(fill="both", expand=True)

        ch_entry_frame = ttk.Frame(ch_frame)
        ch_entry_frame.pack(fill="x", pady=(6, 0))
        self.ch_muscle_entry = ttk.Entry(ch_entry_frame, width=16, font=("Helvetica", 11))
        self.ch_muscle_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.ch_muscle_entry.insert(0, "muscle name")
        self.ch_muscle_entry.bind("<FocusIn>", lambda e: (
            self.ch_muscle_entry.delete(0, "end") if self.ch_muscle_entry.get() == "muscle name" else None
        ))
        ttk.Button(ch_entry_frame, text="Set", command=self._set_channel_label).pack(side="left", padx=2)
        ttk.Button(ch_entry_frame, text="Remove", command=self._remove_channel_label).pack(side="left")

        # Existing users list
        user_frame = ttk.LabelFrame(lists_frame, text="Existing Users", padding=10)
        user_frame.grid(row=0, column=1, padx=4, sticky="nsew")
        self.home_user_listbox = tk.Listbox(user_frame, font=("Helvetica", 12), height=6)
        self.home_user_listbox.pack(fill="both", expand=True)

        # Existing models list
        model_frame = ttk.LabelFrame(lists_frame, text="Trained Models", padding=10)
        model_frame.grid(row=0, column=2, padx=(8, 0), sticky="nsew")
        self.home_model_listbox = tk.Listbox(model_frame, font=("Helvetica", 12), height=6,
                                             exportselection=False)
        self.home_model_listbox.pack(fill="both", expand=True)
        ttk.Button(model_frame, text="Delete Selected Model",
                   command=self._delete_home_model).pack(fill="x", pady=(6, 0))

    def _refresh_known_users(self):
        known = sorted(list_available_users("single"))
        cur = self.user_name_var.get().strip()
        if cur and cur not in known:
            known.append(cur)
        self.user_combo["values"] = known
        # Update home page user listbox
        if hasattr(self, "home_user_listbox"):
            self.home_user_listbox.delete(0, "end")
            for u in known:
                self.home_user_listbox.insert("end", u)

    def _refresh_home_models(self):
        if not hasattr(self, "home_model_listbox"):
            return
        self.home_model_listbox.delete(0, "end")
        self._home_model_entries = []  # (name, mode) for deletion
        seen = set()
        for ch in CHANNEL_NAMES:
            try:
                for name in list_saved_models("single", channel=ch):
                    key = (name, "single")
                    if key not in seen:
                        seen.add(key)
                        self.home_model_listbox.insert("end", name)
                        self._home_model_entries.append(key)
            except Exception:
                pass
        if not self._home_model_entries:
            self.home_model_listbox.insert("end", "(no models trained yet)")

    def _delete_home_model(self):
        sel = self.home_model_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select model", "Click a model in the list first.")
            return
        idx = sel[0]
        if idx >= len(getattr(self, "_home_model_entries", [])):
            return
        name, mode = self._home_model_entries[idx]
        if not messagebox.askyesno("Confirm delete",
                                    f"Delete model '{name}' ({mode})?\n\n"
                                    "This will remove the .joblib file(s) from disk."):
            return
        deleted = delete_named_model(mode, name)
        self._log(f"Deleted model '{name}' ({mode}): {deleted} file(s) removed.")
        self._refresh_home_models()
        messagebox.showinfo("Deleted", f"Model '{name}' deleted ({deleted} file(s)).")

    def _refresh_channel_labels(self):
        if not hasattr(self, "ch_label_listbox"):
            return
        self.ch_label_listbox.delete(0, "end")
        labels = get_channel_labels()
        for ch in CHANNEL_NAMES:
            name = labels.get(ch, "")
            if name:
                self.ch_label_listbox.insert("end", f"{ch}  ->  {name}")
            else:
                self.ch_label_listbox.insert("end", f"{ch}  (not labeled)")

    def _set_channel_label(self):
        sel = self.ch_label_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select channel", "Click a channel in the list first.")
            return
        ch = CHANNEL_NAMES[sel[0]] if sel[0] < len(CHANNEL_NAMES) else None
        if ch is None:
            return
        name = self.ch_muscle_entry.get().strip()
        if not name or name == "muscle name":
            messagebox.showinfo("Enter name", "Type a muscle name in the entry box.")
            return
        try:
            set_channel_label(ch, name)
            self._log(f"Channel {ch} labeled as '{name}'")
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return
        self._refresh_channel_labels()

    def _remove_channel_label(self):
        sel = self.ch_label_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select channel", "Click a channel in the list first.")
            return
        ch = CHANNEL_NAMES[sel[0]] if sel[0] < len(CHANNEL_NAMES) else None
        if ch is None:
            return
        remove_channel_label(ch)
        self._log(f"Channel {ch} label removed")
        self._refresh_channel_labels()

    # ---------------------------------------------------------------
    # TRAINING MENU PAGE (choose Single / Guided)
    # ---------------------------------------------------------------
    def _build_training_menu_page(self):
        page = ttk.Frame(self.container, padding=30)
        self.pages["training_menu"] = page

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 20))
        ttk.Button(header, text="< Back", command=lambda: self._show_page("home")).pack(side="left")
        ttk.Label(header, text="Data Collection", font=("Helvetica", 22, "bold")).pack(side="left", padx=20)

        desc = ttk.Label(page, text="Choose a data collection mode.",
                         font=("Helvetica", 14))
        desc.pack(pady=(10, 30))

        btn_frame = tk.Frame(page)
        btn_frame.pack(expand=True)
        for i in range(2):
            btn_frame.columnconfigure(i, weight=1)

        self._make_card_button(btn_frame, "Single Gesture\nCollection",
                               lambda: self._start_collection("single"),
                               font=("Helvetica", 14, "bold")).grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self._make_card_button(btn_frame, "Guided\nSession",
                               lambda: self._start_collection("session"),
                               font=("Helvetica", 14, "bold")).grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

    # ---------------------------------------------------------------
    # COLLECTION PAGE (3-column: controls | plot | data)
    # ---------------------------------------------------------------
    def _build_collection_page(self):
        page = ttk.Frame(self.container)
        self.pages["collection"] = page

        # top bar: back + user + status + banner
        top = ttk.Frame(page, padding=(12, 8))
        top.pack(fill="x")
        ttk.Button(top, text="< Back", command=self._stop_collection).pack(side="left")
        self.collection_user_label = ttk.Label(top, text="", font=("Helvetica", 14, "bold"))
        self.collection_user_label.pack(side="left", padx=20)
        self.collection_mode_label = ttk.Label(top, text="", font=("Helvetica", 12))
        self.collection_mode_label.pack(side="left", padx=10)
        ttk.Label(top, textvariable=self.status_var, font=("Helvetica", 11)).pack(side="right", padx=10)
        ttk.Button(top, text="Calibrate Idle", command=self._run_calibration).pack(side="right", padx=4)
        ttk.Label(top, textvariable=self.calibration_status_var,
                  font=("Helvetica", 10)).pack(side="right", padx=4)

        # banner — large centered status display with fixed-position timer
        banner_frame = tk.Frame(page, bg="#0f172a", padx=14, pady=14)
        banner_frame.pack(fill="x")
        self.banner_frame_widget = banner_frame

        # Top row: [spacer] [instruction text, centered] [timer, fixed right]
        banner_top = tk.Frame(banner_frame, bg="#0f172a")
        banner_top.pack(fill="x")
        self.banner_top_frame = banner_top

        timer_font = ("Courier", 28, "bold")
        self.banner_timer_label = tk.Label(banner_top, textvariable=self.banner_timer_var,
                                           font=timer_font, bg="#0f172a", fg="#fbbf24",
                                           width=7, anchor="e")
        self.banner_timer_label.pack(side="right")

        self.banner_spacer = tk.Label(banner_top, text="", bg="#0f172a",
                                      font=timer_font, width=7)
        self.banner_spacer.pack(side="left")

        self.banner_label = tk.Label(banner_top, textvariable=self.banner_var,
                                     font=("Helvetica", 30, "bold"), bg="#0f172a", fg="#f8fafc",
                                     anchor="center")
        self.banner_label.pack(side="left", expand=True, fill="x")

        # Bottom row: detail / progress
        self.banner_detail_label = tk.Label(banner_frame, textvariable=self.banner_detail_var,
                                            font=("Helvetica", 16), bg="#0f172a", fg="#cbd5e1",
                                            anchor="center")
        self.banner_detail_label.pack(fill="x", pady=(6, 0))

        # log panel (at bottom)
        self._build_log_panel(page, "collection")

        # 3-column content
        content = ttk.Frame(page)
        content.pack(fill="both", expand=True, padx=8, pady=8)
        content.columnconfigure(1, weight=3)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(2, weight=1)
        content.rowconfigure(0, weight=1)

        # LEFT: controls
        left = ttk.LabelFrame(content, text="Controls", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.controls_parent = left

        # create a canvas for scrolling controls
        ctrl_canvas = tk.Canvas(left, highlightthickness=0, width=280)
        ctrl_scroll = ttk.Scrollbar(left, orient="vertical", command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_scroll.pack(side="right", fill="y")
        ctrl_canvas.pack(side="left", fill="both", expand=True)
        self.controls_holder = ttk.Frame(ctrl_canvas)
        self.controls_window = ctrl_canvas.create_window((0, 0), window=self.controls_holder, anchor="nw")
        self.controls_canvas = ctrl_canvas
        self.controls_holder.bind("<Configure>", lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.bind("<Configure>", lambda e: ctrl_canvas.itemconfigure(self.controls_window, width=e.width))

        def _on_mousewheel(event):
            if event.delta:
                ctrl_canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                ctrl_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                ctrl_canvas.yview_scroll(1, "units")

        ctrl_canvas.bind("<Enter>", lambda e: (
            self.root.bind_all("<MouseWheel>", _on_mousewheel),
            self.root.bind_all("<Button-4>", _on_mousewheel),
            self.root.bind_all("<Button-5>", _on_mousewheel),
        ))
        ctrl_canvas.bind("<Leave>", lambda e: (
            self.root.unbind_all("<MouseWheel>"),
            self.root.unbind_all("<Button-4>"),
            self.root.unbind_all("<Button-5>"),
        ))

        # CENTER: plot
        center = ttk.LabelFrame(content, text="Live EMG Signal", padding=8)
        center.grid(row=0, column=1, sticky="nsew", padx=6)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.plot_axis = self.figure.subplots(1, 1)
        for ch in CHANNEL_NAMES:
            line, = self.plot_axis.plot([], [], linewidth=1.5, color=PLOT_COLORS.get(ch), label=channel_display_name(ch))
            self.lines_by_channel[ch] = line
        self.plot_axis.set_xlabel("Recent samples")
        self.plot_axis.set_ylabel("ADC")
        self.plot_axis.set_ylim(0, ADC_MAX)
        self.plot_axis.grid(True)
        self.plot_axis.legend(loc="upper left")
        self.figure.subplots_adjust(top=0.92, right=0.98, left=0.10, bottom=0.11)
        self.canvas = FigureCanvasTkAgg(self.figure, master=center)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # RIGHT: existing data
        right = ttk.LabelFrame(content, text="Collected Data", padding=8)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        self.data_summary_var = tk.StringVar(value="No data yet.")
        ttk.Label(right, textvariable=self.data_summary_var, wraplength=260).pack(anchor="w", pady=(0, 8))

        columns = ("label", "count", "mode", "channel")
        self.capture_tree = ttk.Treeview(right, columns=columns, show="headings", height=18)
        self.capture_tree.heading("label", text="Gesture")
        self.capture_tree.heading("count", text="#")
        self.capture_tree.heading("mode", text="Mode")
        self.capture_tree.heading("channel", text="Channel")
        self.capture_tree.column("label", width=90, stretch=True)
        self.capture_tree.column("count", width=35, stretch=False)
        self.capture_tree.column("mode", width=70, stretch=False)
        self.capture_tree.column("channel", width=55, stretch=False)
        tree_scroll = ttk.Scrollbar(right, orient="vertical", command=self.capture_tree.yview)
        self.capture_tree.configure(yscrollcommand=tree_scroll.set)
        self.capture_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="left", fill="y")

        ttk.Button(right, text="Refresh", command=self._refresh_capture_list).pack(fill="x", pady=(8, 0))
        ttk.Button(right, text="Clear User Data", command=self._clear_user_data).pack(fill="x", pady=(4, 0))

    # ---------------------------------------------------------------
    # TRAINING PAGE (model building)
    # ---------------------------------------------------------------
    def _build_training_page(self):
        page = ttk.Frame(self.container, padding=16)
        self.pages["model_training"] = page

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 12))
        ttk.Button(header, text="< Back", command=lambda: self._show_page("home")).pack(side="left")
        ttk.Label(header, text="Model Training", font=("Helvetica", 20, "bold")).pack(side="left", padx=20)
        ttk.Button(header, text="Refresh All", command=self._refresh_all_training).pack(side="right")

        # log panel (at bottom)
        self._build_log_panel(page, "model_training")

        form = ttk.Frame(page)
        form.pack(fill="both", expand=True)
        form.columnconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, weight=2)
        form.rowconfigure(0, weight=1)

        # ---- LEFT: Train Base Model ----
        left = ttk.LabelFrame(form, text="Train Base Model", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        ttk.Label(left, text="Train on all selected users' data.\n"
                  "This is the shared base model.",
                  wraplength=220, justify="left", font=("Helvetica", 10)).pack(anchor="w", pady=(0, 8))

        base_form = ttk.Frame(left)
        base_form.pack(fill="x")
        base_form.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(base_form, text="Mode").grid(row=r, column=0, sticky="w", pady=3)
        mode_cb = ttk.Combobox(base_form, textvariable=self.training_mode_var, state="readonly",
                     values=("single",), width=16)
        mode_cb.grid(row=r, column=1, sticky="we", pady=3)
        mode_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_all_training())

        r += 1
        ttk.Label(base_form, text="Model name").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(base_form, textvariable=self.model_name_var, width=16).grid(row=r, column=1, sticky="we", pady=3)

        r += 1
        ttk.Label(base_form, text="Model type").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Combobox(base_form, textvariable=self.training_model_type_var, state="readonly",
                     values=tuple(model_type_labels()), width=16).grid(row=r, column=1, sticky="we", pady=3)

        r += 1
        ttk.Label(base_form, text="Channels").grid(row=r, column=0, sticky="nw", pady=(8, 2))
        r += 1
        self.home_train_channel_listbox = tk.Listbox(base_form, selectmode=tk.MULTIPLE, exportselection=False, height=3)
        self.home_train_channel_listbox.grid(row=r, column=0, columnspan=2, sticky="we", pady=2)
        self.home_train_channel_listbox.bind("<<ListboxSelect>>", lambda e: self._on_train_channel_changed())

        r += 1
        ttk.Label(base_form, text="Users").grid(row=r, column=0, sticky="nw", pady=(8, 2))
        r += 1
        self.home_train_user_listbox = tk.Listbox(base_form, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.home_train_user_listbox.grid(row=r, column=0, columnspan=2, sticky="we", pady=2)
        self.home_train_user_listbox.bind("<<ListboxSelect>>", lambda e: self._on_train_user_changed())

        r += 1
        ttk.Label(base_form, text="Labels").grid(row=r, column=0, sticky="nw", pady=(8, 2))
        r += 1
        self.home_train_label_listbox = tk.Listbox(base_form, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.home_train_label_listbox.grid(row=r, column=0, columnspan=2, sticky="we", pady=2)

        self._make_card_button(left, "Train Base Model", self._train_model,
                               font=("Helvetica", 14, "bold"), pady=10).pack(fill="x", pady=(12, 0))

        # ---- MIDDLE: Model Management ----
        mid = ttk.LabelFrame(form, text="Model Management", padding=12)
        mid.grid(row=0, column=1, sticky="nsew", padx=6)

        ttk.Label(mid, text="Continue from a saved base model with the selected users' data.\n"
                  "A new model is trained and saved separately.",
                  wraplength=220, justify="left", font=("Helvetica", 10)).pack(anchor="w", pady=(0, 12))

        ft_form = ttk.Frame(mid)
        ft_form.pack(fill="x")
        ft_form.columnconfigure(1, weight=1)

        ttk.Label(ft_form, text="Saved model").grid(row=0, column=0, sticky="w", pady=3)
        self.ft_base_var = tk.StringVar()
        self.ft_base_combo = ttk.Combobox(ft_form, textvariable=self.ft_base_var, state="readonly", width=16)
        self.ft_base_combo.grid(row=0, column=1, sticky="we", pady=3)
        self.ft_base_combo.bind("<<ComboboxSelected>>", lambda e: self._on_finetune_base_changed())

        ttk.Label(ft_form, text="New model").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(ft_form, textvariable=self.ft_new_model_name_var, width=16).grid(row=1, column=1, sticky="we", pady=3)

        ttk.Button(mid, text="Continue Training", command=self._run_finetune).pack(fill="x", pady=(12, 0))
        ttk.Button(mid, text="Delete Selected Model", command=self._delete_training_model).pack(fill="x", pady=(12, 0))

        # ---- RIGHT: Training Report ----
        right = ttk.LabelFrame(form, text="Training Report", padding=12)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.training_report_text = tk.Text(right, wrap="word", font=("Courier", 11),
                                            state="disabled", bg="#fafafa")
        report_scroll = ttk.Scrollbar(right, orient="vertical", command=self.training_report_text.yview)
        self.training_report_text.configure(yscrollcommand=report_scroll.set)
        self.training_report_text.pack(side="left", fill="both", expand=True)
        report_scroll.pack(side="right", fill="y")

    # ---------------------------------------------------------------
    # TESTING PAGE
    # ---------------------------------------------------------------
    def _build_testing_page(self):
        page = ttk.Frame(self.container, padding=16)
        self.pages["testing"] = page

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 12))
        ttk.Button(header, text="< Back", command=self._stop_testing).pack(side="left")
        ttk.Label(header, text="Live Testing", font=("Helvetica", 20, "bold")).pack(side="left", padx=20)
        ttk.Label(header, textvariable=self.status_var, font=("Helvetica", 11)).pack(side="right")

        # log panel (at bottom)
        self._build_log_panel(page, "testing")

        content = ttk.Frame(page)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=3)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        # left: model selection + prediction
        left = ttk.LabelFrame(content, text="Model & Prediction", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(left, text="Mode").grid(row=0, column=0, sticky="w", pady=4)
        mode_cb = ttk.Combobox(left, textvariable=self.testing_mode_var, state="readonly",
                               values=("single",), width=18)
        mode_cb.grid(row=0, column=1, sticky="w", pady=4)
        mode_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_test_models())

        ttk.Label(left, text="Model").grid(row=1, column=0, sticky="w", pady=4)
        self.model_combo = ttk.Combobox(left, textvariable=self.model_choice_var, state="readonly", width=22)
        self.model_combo.grid(row=1, column=1, sticky="we", pady=4)

        self.test_model_info_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.test_model_info_var, font=("Helvetica", 10),
                  foreground="#666").grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        btn_row = ttk.Frame(left)
        btn_row.grid(row=3, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Button(btn_row, text="Refresh", command=self._refresh_test_models).pack(side="left")
        ttk.Button(btn_row, text="Run Test", command=self._run_test).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Stop Test", command=self._stop_predictor).pack(side="left", padx=(8, 0))
        btn_row2 = ttk.Frame(left)
        btn_row2.grid(row=4, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Checkbutton(
            btn_row2,
            text="Debug Mode",
            variable=self.debug_mode_var,
            command=self._on_debug_mode_toggle,
        ).pack(side="left")
        ttk.Button(btn_row2, text="Export Debug Report", command=self._export_debug_report).pack(side="left", padx=(8, 0))

        # Calibration
        ttk.Separator(left, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="we", pady=(10, 6))

        cal_row = ttk.Frame(left)
        cal_row.grid(row=6, column=0, columnspan=2, sticky="we")
        ttk.Button(cal_row, text="Calibrate Idle", command=self._run_calibration).pack(side="left")
        ttk.Label(cal_row, textvariable=self.calibration_status_var,
                  font=("Helvetica", 10), foreground="#666").pack(side="left", padx=(8, 0))

        # Confidence threshold slider
        ttk.Label(left, text="Confidence threshold").grid(row=7, column=0, sticky="w", pady=(6, 0))
        thresh_row = ttk.Frame(left)
        thresh_row.grid(row=8, column=0, columnspan=2, sticky="we")
        ttk.Scale(thresh_row, from_=0.0, to=1.0, variable=self.confidence_threshold_var,
                  orient="horizontal", command=self._on_threshold_change).pack(side="left", fill="x", expand=True)
        ttk.Label(thresh_row, textvariable=self.threshold_display_var,
                  font=("Courier", 11), width=5).pack(side="left", padx=(6, 0))

        ttk.Separator(left, orient="horizontal").grid(row=9, column=0, columnspan=2, sticky="we", pady=10)

        ttk.Label(left, text="Predicted:", font=("Helvetica", 14, "bold")).grid(row=10, column=0, sticky="w")
        self.prediction_display_label = ttk.Label(left, textvariable=self.prediction_display_var,
                  font=("Helvetica", 28, "bold"), foreground="#0f766e",
                  wraplength=250)
        self.prediction_display_label.grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 10))
        ttk.Label(left, textvariable=self.prediction_var, wraplength=280,
                  justify="left").grid(row=12, column=0, columnspan=2, sticky="w")

        ttk.Separator(left, orient="horizontal").grid(row=12, column=0, columnspan=2, sticky="we", pady=10)

        hist_header = ttk.Frame(left)
        hist_header.grid(row=13, column=0, columnspan=2, sticky="we", pady=(0, 4))
        ttk.Label(hist_header, text="Prediction History", font=("Helvetica", 12, "bold")).pack(side="left")
        ttk.Button(hist_header, text="Clear", command=self._clear_prediction_history).pack(side="right")

        hist_frame = ttk.Frame(left)
        hist_frame.grid(row=14, column=0, columnspan=2, sticky="nsew")
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.rowconfigure(0, weight=1)
        self.prediction_history_tree = ttk.Treeview(
            hist_frame,
            columns=("time", "gesture", "confidence"),
            show="headings",
            height=8,
        )
        self.prediction_history_tree.heading("time", text="Time")
        self.prediction_history_tree.heading("gesture", text="Gesture")
        self.prediction_history_tree.heading("confidence", text="Conf")
        self.prediction_history_tree.column("time", width=72, stretch=False)
        self.prediction_history_tree.column("gesture", width=110, stretch=True)
        self.prediction_history_tree.column("confidence", width=54, stretch=False, anchor="e")
        hist_scroll = ttk.Scrollbar(hist_frame, orient="vertical", command=self.prediction_history_tree.yview)
        self.prediction_history_tree.configure(yscrollcommand=hist_scroll.set)
        self.prediction_history_tree.grid(row=0, column=0, sticky="nsew")
        hist_scroll.grid(row=0, column=1, sticky="ns")

        left.columnconfigure(1, weight=1)
        left.rowconfigure(14, weight=1)

        # right: plot
        right = ttk.LabelFrame(content, text="Live EMG Signal", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        self.test_figure = Figure(figsize=(7, 5), dpi=100)
        self.test_plot_axis = self.test_figure.subplots(1, 1)
        self.test_lines = {}
        for ch in CHANNEL_NAMES:
            line, = self.test_plot_axis.plot([], [], linewidth=1.5, color=PLOT_COLORS.get(ch), label=channel_display_name(ch))
            self.test_lines[ch] = line
        self.test_plot_axis.set_xlabel("Recent samples")
        self.test_plot_axis.set_ylabel("ADC")
        self.test_plot_axis.set_ylim(0, ADC_MAX)
        self.test_plot_axis.grid(True)
        self.test_plot_axis.legend(loc="upper left")
        self.test_figure.subplots_adjust(top=0.92, right=0.98, left=0.10, bottom=0.11)
        self.test_canvas = FigureCanvasTkAgg(self.test_figure, master=right)
        self.test_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------------------------------------------------------
    # Navigation actions
    # ---------------------------------------------------------------
    def _go_training_menu(self):
        if not self.user_name_var.get().strip():
            messagebox.showerror("Missing user", "Please enter a user name first.")
            return
        self._show_page("training_menu")

    def _go_testing(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Missing port", "Select a serial port first.")
            return
        if serial is None:
            messagebox.showerror("Missing dependency", f"pyserial is required.\n\n{SERIAL_IMPORT_ERROR}")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.01)
        except SERIAL_EXCEPTION as exc:
            messagebox.showerror("Serial error", str(exc))
            return
        self.serial_ready_at = time.time() + ARDUINO_WARMUP_SECONDS
        self.session_mode = "test"
        self.status_var.set("Warming up")
        self._reset_plot_buffers()
        self._refresh_test_models()
        # Auto-load calibration for current user
        user = self.user_name_var.get().strip()
        self.calibration_data = load_calibration(user) if user else None
        self._update_calibration_display()
        self._show_page("testing")
        cal_info = "calibrated" if self.calibration_data else "no calibration"
        self._log(f"Testing session started on port {port}. ({cal_info})")

    def _go_model_training(self):
        self._refresh_all_training()
        self._show_page("model_training")

    def _start_collection(self, mode):
        if serial is None:
            messagebox.showerror("Missing dependency", f"pyserial is required.\n\n{SERIAL_IMPORT_ERROR}")
            return
        raw_user = self.user_name_var.get().strip()
        if not raw_user:
            messagebox.showerror("Missing user", "Enter a user name first.")
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Missing port", "Select a serial port first.")
            return
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.01)
        except SERIAL_EXCEPTION as exc:
            messagebox.showerror("Serial error", str(exc))
            return

        self.session_user = sanitize_user_name(raw_user)
        self.session_mode = mode
        self.serial_ready_at = time.time() + ARDUINO_WARMUP_SECONDS
        self.detected_channels = {DEFAULT_CHANNEL}
        self._reset_capture_state()
        self._reset_plot_buffers()
        self._reset_guided_state()

        # Auto-load calibration
        self.calibration_data = load_calibration(self.session_user)
        self._update_calibration_display()

        if mode == "single":
            labels = list_labels_for_mode("single")
            if labels:
                self.selected_label_var.set(labels[0] if labels[0] != "rest" else (labels[1] if len(labels) > 1 else labels[0]))
        else:
            labels = self._guided_label_options()
            if labels:
                self.selected_label_var.set(labels[0])

        self.collection_user_label.configure(text=f"User: {self.session_user}")
        mode_text = {"single": "Single Gesture", "session": "Guided Session"}.get(mode, mode)
        self.collection_mode_label.configure(text=f"Mode: {mode_text}")
        self.status_var.set("Warming up")
        self.banner_var.set("WAITING FOR ARDUINO RESET...")
        self.banner_timer_var.set("")
        self.banner_detail_var.set("")

        self._render_controls()
        self._refresh_capture_list()
        self._show_page("collection")
        self.root.bind_all("<Key>", self._on_key_press)
        self._log(f"Collection started: user={self.session_user}, mode={mode_text}, port={port}.")

    def _stop_collection(self):
        self._reset_capture_state()
        self._reset_guided_state()
        self._reset_plot_buffers()
        self.session_user = None
        self.session_mode = None
        self.status_var.set("Idle")
        self._close_serial()
        self.root.unbind_all("<Key>")
        self._refresh_known_users()
        self._show_page("training_menu")

    def _stop_testing(self):
        self._stop_predictor()
        self._reset_plot_buffers()
        self.session_mode = None
        self.status_var.set("Idle")
        self._close_serial()
        self._show_page("home")

    def _close_serial(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except SERIAL_EXCEPTION:
                pass
            self.ser = None

    # ---------------------------------------------------------------
    # Controls rendering (left panel of collection page)
    # ---------------------------------------------------------------
    def _render_controls(self):
        for child in self.controls_holder.winfo_children():
            child.destroy()
        self.guided_session_labels_listbox = None

        if self.session_mode == "single":
            self._render_single_controls()
        elif self.session_mode == "session":
            self._render_guided_controls()
        self.controls_canvas.yview_moveto(0.0)

    def _render_label_section(self, parent):
        box = ttk.LabelFrame(parent, text="Gestures", padding=8)
        box.pack(fill="x", pady=(0, 10))
        labels = list_labels_for_mode("single")
        if self.session_mode == "session":
            labels = self._guided_label_options()
        for label in labels:
            ttk.Radiobutton(box, text=label, value=label, variable=self.selected_label_var).pack(anchor="w")
        add_row = ttk.Frame(box)
        add_row.pack(fill="x", pady=(8, 0))
        ttk.Entry(add_row, textvariable=self.new_label_var, width=14).pack(side="left", fill="x", expand=True)
        ttk.Button(add_row, text="+", width=3, command=self._add_label).pack(side="left", padx=(4, 0))
        ttk.Button(add_row, text="-", width=3, command=self._delete_label).pack(side="left", padx=(4, 0))

    def _render_single_controls(self):
        parent = self.controls_holder
        self._render_label_section(parent)

        frame = ttk.LabelFrame(parent, text="Single Capture", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text="Captures per run").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.single_auto_count_var, width=6).grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(frame, text="Capture seconds").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=0.5, to=15.0, increment=0.5, textvariable=self.single_window_seconds_var, width=6).grid(row=1, column=1, sticky="w", pady=3)

        ttk.Label(frame, text="Prepare seconds").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=0.0, to=10.0, increment=0.5, textvariable=self.prepare_seconds_var, width=6).grid(row=2, column=1, sticky="w", pady=3)

        ttk.Button(frame, text="Start Capture", command=self._start_single_manual).grid(row=3, column=0, sticky="we", pady=(8, 2))
        ttk.Button(frame, text="Auto Run", command=self._start_single_auto).grid(row=3, column=1, sticky="we", padx=(4, 0), pady=(8, 2))
        ttk.Button(frame, text="Cancel", command=self._cancel_auto).grid(row=4, column=0, columnspan=2, sticky="we", pady=2)

    def _render_guided_controls(self):
        parent = self.controls_holder
        self._render_label_section(parent)

        frame = ttk.LabelFrame(parent, text="Guided Session", padding=8)
        frame.pack(fill="x", pady=(0, 10))

        ttk.Label(frame, text="Labels to collect").grid(row=0, column=0, sticky="w", pady=(0, 4))
        label_box = ttk.Frame(frame)
        label_box.grid(row=1, column=0, columnspan=2, sticky="we")
        self.guided_session_labels_listbox = tk.Listbox(label_box, selectmode=tk.MULTIPLE, exportselection=False, height=5)
        self.guided_session_labels_listbox.pack(side="left", fill="both", expand=True)
        self._refresh_guided_labels()

        ttk.Label(frame, text="Single repeats/label").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.guided_single_repeats_var, width=6).grid(row=2, column=1, sticky="w", pady=3)

        ttk.Label(frame, text="Single capture seconds").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=0.5, to=15.0, increment=0.5, textvariable=self.guided_single_seconds_var, width=6).grid(row=3, column=1, sticky="w", pady=3)

        ttk.Label(frame, text="Single rest seconds").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Spinbox(frame, from_=0.0, to=15.0, increment=0.5, textvariable=self.guided_single_rest_seconds_var, width=6).grid(row=4, column=1, sticky="w", pady=3)

        ttk.Checkbutton(frame, text="Break between gestures", variable=self.guided_break_enabled_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=3)

        btn = ttk.Frame(frame)
        btn.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Button(btn, text="Auto Run", command=self._start_guided_auto).pack(side="left", fill="x", expand=True)
        ttk.Button(btn, text="Manual Step", command=self._start_guided_step).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(frame, text="Cancel Session", command=self._cancel_guided).grid(row=7, column=0, columnspan=2, sticky="we", pady=(4, 0))

    # ---------------------------------------------------------------
    # Label management
    # ---------------------------------------------------------------
    def _target_modes(self):
        if self.session_mode in ("session", "single"):
            return ["single"]
        return []

    def _add_label(self):
        raw = self.new_label_var.get()
        for mode in self._target_modes():
            try:
                add_label_to_mode(mode, raw)
            except ValueError as exc:
                self._log(f"Add label failed: {exc}", level="ERROR")
                messagebox.showerror("Invalid label", str(exc))
                return
        self._log(f"Label added: {raw} to {self._target_modes()}")
        self.new_label_var.set("")
        self._render_controls()

    def _delete_label(self):
        label = self.selected_label_var.get().strip()
        if not label:
            return
        for mode in self._target_modes():
            if label in list_labels_for_mode(mode):
                try:
                    remove_label_from_mode(mode, label)
                except ValueError as exc:
                    self._log(f"Delete label failed: {exc}", level="ERROR")
                    messagebox.showerror("Cannot delete", str(exc))
                    return
        self._log(f"Label deleted: {label}")
        self._render_controls()

    # ---------------------------------------------------------------
    # Guided session helpers
    # ---------------------------------------------------------------
    def _guided_label_options(self):
        seen = set()
        labels = []
        for label in list_labels_for_mode("single"):
            if label == "rest" or label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    def _get_guided_selected_labels(self):
        if self.guided_session_labels_listbox is None:
            return []
        return [self.guided_session_labels_listbox.get(i) for i in self.guided_session_labels_listbox.curselection()]

    def _refresh_guided_labels(self):
        if self.guided_session_labels_listbox is None:
            return
        self.guided_session_labels_listbox.delete(0, tk.END)
        for label in self._guided_label_options():
            self.guided_session_labels_listbox.insert(tk.END, label)
        self.guided_session_labels_listbox.selection_set(0, tk.END)

    def _build_guided_tasks(self, labels):
        single_reps = max(1, int(self.guided_single_repeats_var.get()))
        single_sec = max(0.5, float(self.guided_single_seconds_var.get()))
        single_rest_s = max(0.0, float(self.guided_single_rest_seconds_var.get()))
        insert_breaks = self.guided_break_enabled_var.get()

        # ---- build gesture groups ----
        # Each group = one gesture, captured N times with optional rest between reps
        gesture_groups = []

        for label in labels:
            group_tasks = []
            for i in range(single_reps):
                group_tasks.append({
                    "capture_mode": "single", "label": label,
                    "duration": single_sec,
                })
                if single_rest_s > 0 and i < single_reps - 1:
                    group_tasks.append({
                        "capture_mode": "single", "label": "rest",
                        "duration": single_rest_s, "no_save": True,
                        "no_prepare": True,
                    })
            gesture_groups.append({
                "name": label, "mode": "single",
                "tasks": group_tasks, "collection_total": single_reps,
            })

        # ---- flatten with breaks and metadata ----
        gesture_total = len(gesture_groups)
        tasks = []
        for gi, group in enumerate(gesture_groups):
            # Insert break between gesture groups
            if gi > 0 and insert_breaks:
                tasks.append({
                    "capture_mode": "break",
                    "label": "break",
                    "next_gesture_name": group["name"],
                    "next_gesture_mode": group["mode"],
                    "next_gesture_index": gi + 1,
                    "gesture_total": gesture_total,
                    "next_collection_total": group["collection_total"],
                })

            collection_idx = 0
            for task in group["tasks"]:
                # Only actual gesture captures (not rest) start a new collection cycle
                if task["label"] != "rest":
                    collection_idx += 1

                task.update({
                    "gesture_name": group["name"],
                    "gesture_mode": group["mode"],
                    "gesture_index": gi + 1,
                    "gesture_total": gesture_total,
                    "collection_index": collection_idx,
                    "collection_total": group["collection_total"],
                })
                tasks.append(task)

        return tasks

    def _reset_guided_state(self):
        self.guided_session_active = False
        self.guided_session_auto = False
        self.guided_session_waiting = False
        self.guided_session_paused = False
        self.guided_session_tasks = []
        self.guided_session_step_index = 0
        self.guided_break_state = None
        self._last_flash_label = None

    # ---------------------------------------------------------------
    # Capture actions
    # ---------------------------------------------------------------
    def _reset_capture_state(self):
        self.prepare_state = None
        self.active_capture = None
        self.auto_running = False
        self.auto_queue = []

    def _reset_plot_buffers(self):
        for ch in CHANNEL_NAMES:
            self.channel_buffers[ch]["time"].clear()
            self.channel_buffers[ch]["adc"].clear()
            self.channel_buffers[ch]["voltage"].clear()
        for ch, line in self.lines_by_channel.items():
            line.set_data([], [])
        # reset Y-axis to full hardware range so expand-only starts fresh
        if hasattr(self, "plot_axis"):
            self.plot_axis.set_ylim(0, ADC_MAX)
        if hasattr(self, "test_lines"):
            for ch, line in self.test_lines.items():
                line.set_data([], [])
        if hasattr(self, "test_plot_axis"):
            self.test_plot_axis.set_ylim(0, ADC_MAX)
        self._plot_dirty = True
        self._last_plot_refresh_at = 0.0

    def _current_channels(self):
        return [ch for ch in CHANNEL_NAMES if ch in self.detected_channels] or [DEFAULT_CHANNEL]

    def _start_prepare(self, label, duration, auto=False, capture_mode=None, guided=False, no_save=False, no_prepare=False):
        prep = 0.0 if no_prepare else max(0.0, float(self.prepare_seconds_var.get()))
        self.prepare_state = {
            "label": label, "duration": float(duration), "auto": auto,
            "capture_mode": capture_mode or self.session_mode,
            "guided": guided, "no_save": no_save, "deadline": time.time() + prep,
        }
        self.status_var.set("Prepare")
        self._log(f"Preparing capture: label={label}, duration={duration:.1f}s, prep={prep:.1f}s")

    def _begin_capture(self, label, duration, auto, capture_mode=None, guided=False, no_save=False):
        self.prepare_state = None
        self.active_capture = {
            "label": label, "duration": float(duration), "auto": auto,
            "capture_mode": capture_mode or self.session_mode,
            "guided": guided, "no_save": no_save, "started_at": time.time(),
            "rows_by_channel": {ch: [] for ch in self._current_channels()},
        }
        self.status_var.set("Capturing")
        self._log(f"Capture started: label={label}, duration={duration:.1f}s, channels={self._current_channels()}")

    def _finish_capture(self):
        cap = self.active_capture
        self.active_capture = None
        if cap is None or not self.session_user:
            return

        if not cap.get("no_save"):
            channels_to_save = [ch for ch, rows in cap["rows_by_channel"].items() if len(rows) >= 5]
            if channels_to_save:
                shared_idx = next_shared_capture_index(self.session_user, cap["capture_mode"], cap["label"], channels_to_save)
                shared_fn = f"{self.session_user}_emg_{cap['label']}_{shared_idx:03d}.csv"
                for ch in channels_to_save:
                    write_capture(cap["rows_by_channel"][ch], self.session_user, cap["capture_mode"], cap["label"], ch, shared_fn)
                self._refresh_capture_list()
                total_rows = sum(len(cap["rows_by_channel"][ch]) for ch in channels_to_save)
                self._log(f"Saved capture: label={cap['label']}, channels={channels_to_save}, samples={total_rows}, file={shared_fn}")
            else:
                self._log(f"Capture skipped (too few samples): label={cap['label']}", level="WARN")

        if cap.get("guided"):
            self.guided_session_step_index += 1

        if cap["auto"] and self.auto_queue:
            self.auto_queue.pop(0)
        if cap["auto"] and self.auto_queue:
            self._queue_task(self.auto_queue[0], auto=True)
            return

        if cap.get("guided") and self.guided_session_active:
            if self.guided_session_step_index >= len(self.guided_session_tasks):
                self._reset_capture_state()
                self._reset_guided_state()
                self.status_var.set("Finished")
                self.banner_var.set("GUIDED SESSION COMPLETE")
                self.banner_timer_var.set("")
                self._log("Guided session completed successfully.")
                return
            if not cap["auto"]:
                # In manual mode, auto-start breaks so the user doesn't have to
                next_task = self.guided_session_tasks[self.guided_session_step_index]
                if next_task.get("capture_mode") == "break":
                    self._queue_task(dict(next_task, guided=True), auto=False)
                else:
                    self.guided_session_waiting = True
                    self.status_var.set("Ready")
                return

        self.auto_running = False
        self.status_var.set("Saved")

    def _finish_break(self):
        brk = self.guided_break_state
        self.guided_break_state = None
        if brk is None:
            return
        auto = brk["auto"]

        # advance step index (break counts as a task in the list)
        if self.guided_session_active:
            self.guided_session_step_index += 1

        # advance auto_queue
        if auto and self.auto_queue:
            self.auto_queue.pop(0)
        if auto and self.auto_queue:
            self._queue_task(self.auto_queue[0], auto=True)
            return

        if self.guided_session_active:
            if self.guided_session_step_index >= len(self.guided_session_tasks):
                self._reset_capture_state()
                self._reset_guided_state()
                self.status_var.set("Finished")
                self.banner_var.set("GUIDED SESSION COMPLETE")
                self.banner_timer_var.set("")
                self._log("Guided session completed (after break).")
                return
            if not auto:
                self.guided_session_waiting = True
                self.status_var.set("Ready")
                return

        self.status_var.set("Ready")

    def _queue_task(self, task, auto=False):
        if task.get("capture_mode") == "break":
            self.guided_break_state = {
                "auto": auto,
                "task": task,
            }
            self.status_var.set("Break")
            self._log(f"Break: press Space to continue to {task.get('next_gesture_name', '?')}")
            return
        self.selected_label_var.set(task["label"])
        self._start_prepare(task["label"], task["duration"], auto=auto,
                           capture_mode=task.get("capture_mode"), guided=task.get("guided", False),
                           no_save=task.get("no_save", False), no_prepare=task.get("no_prepare", False))

    # --- single ---
    def _start_single_manual(self):
        if self.prepare_state or self.active_capture or self.auto_running:
            return
        label = self.selected_label_var.get()
        dur = max(0.5, float(self.single_window_seconds_var.get()))
        self._start_prepare(label, dur, auto=False, capture_mode="single")

    def _start_single_auto(self):
        if self.prepare_state or self.active_capture or self.auto_running:
            return
        count = max(1, int(self.single_auto_count_var.get()))
        label = self.selected_label_var.get()
        dur = float(self.single_window_seconds_var.get())
        self.auto_queue = [{"capture_mode": "single", "label": label, "duration": dur} for _ in range(count)]
        self.auto_running = True
        self._queue_task(self.auto_queue[0], auto=True)

    # --- guided ---
    def _start_guided_auto(self):
        if self.prepare_state or self.active_capture or self.auto_running or self.guided_session_active:
            return
        labels = self._get_guided_selected_labels()
        if not labels:
            messagebox.showerror("No labels", "Select at least one label.")
            return
        tasks = self._build_guided_tasks(labels)
        self.guided_session_tasks = tasks
        self.guided_session_step_index = 0
        self.guided_session_active = True
        self.guided_session_auto = True
        self.auto_queue = [dict(t, guided=True) for t in tasks]
        self.auto_running = True
        self._queue_task(self.auto_queue[0], auto=True)

    def _start_guided_step(self):
        # If in break, Space ends the break and resumes
        if self.guided_break_state is not None:
            self._finish_break()
            return
        if self.prepare_state or self.active_capture or self.auto_running:
            return
        if not self.guided_session_active:
            labels = self._get_guided_selected_labels()
            if not labels:
                messagebox.showerror("No labels", "Select at least one label.")
                return
            self.guided_session_tasks = self._build_guided_tasks(labels)
            self.guided_session_step_index = 0
            self.guided_session_active = True
            self.guided_session_auto = False
            self.guided_session_waiting = True
        if self.guided_session_step_index >= len(self.guided_session_tasks):
            self._reset_guided_state()
            self.status_var.set("Finished")
            return
        task = self.guided_session_tasks[self.guided_session_step_index]
        self.guided_session_waiting = False
        self._queue_task(dict(task, guided=True), auto=False)

    def _cancel_guided(self):
        self._reset_capture_state()
        self._reset_guided_state()
        self.status_var.set("Idle")

    def _cancel_auto(self):
        self._reset_capture_state()
        if self.session_mode == "session":
            self._reset_guided_state()
        self.status_var.set("Idle")

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    def _refresh_training_lists(self):
        mode = self.training_mode_var.get()
        if self.home_train_channel_listbox:
            chs = list_available_channels(mode)
            self._train_channel_ids = chs  # store raw ids for lookup
            self.home_train_channel_listbox.delete(0, tk.END)
            for ch in chs:
                self.home_train_channel_listbox.insert(tk.END, f"{channel_display_name(ch)} ({ch})")
            self.home_train_channel_listbox.selection_set(0, tk.END)
        self._on_train_channel_changed()

    def _get_selected_train_channels(self):
        """Return raw channel IDs for selected items in the training channel list."""
        if not self.home_train_channel_listbox or not hasattr(self, "_train_channel_ids"):
            return None
        indices = self.home_train_channel_listbox.curselection()
        return [self._train_channel_ids[i] for i in indices if i < len(self._train_channel_ids)]

    def _on_train_channel_changed(self):
        mode = self.training_mode_var.get()
        sel_chs = self._get_selected_train_channels()
        users = list_available_users(mode, selected_channels=sel_chs or None)
        if self.home_train_user_listbox:
            self.home_train_user_listbox.delete(0, tk.END)
            for u in users:
                self.home_train_user_listbox.insert(tk.END, u)
            self.home_train_user_listbox.selection_set(0, tk.END)
        self._on_train_user_changed()

    def _on_train_user_changed(self):
        mode = self.training_mode_var.get()
        sel_chs = self._get_selected_train_channels()
        sel_users = [self.home_train_user_listbox.get(i) for i in self.home_train_user_listbox.curselection()] if self.home_train_user_listbox else None
        labels = list_available_labels(mode, selected_users=sel_users, selected_channels=sel_chs)
        if self.home_train_label_listbox:
            self.home_train_label_listbox.delete(0, tk.END)
            for l in labels:
                self.home_train_label_listbox.insert(tk.END, l)
            self.home_train_label_listbox.selection_set(0, tk.END)

    def _set_training_report(self, text):
        self.training_report_text.configure(state="normal")
        self.training_report_text.delete("1.0", "end")
        self.training_report_text.insert("1.0", text)
        self.training_report_text.configure(state="disabled")
        self.training_report_text.see("1.0")
        self.root.update_idletasks()

    def _refresh_all_training(self):
        self._refresh_training_lists()
        self._refresh_finetune_lists()

    def _train_model(self):
        mode = self.training_mode_var.get()
        name = self.model_name_var.get().strip()
        if not name:
            messagebox.showerror("Missing name", "Enter a model name.")
            return
        sel_chs = self._get_selected_train_channels() or []
        sel_users = [self.home_train_user_listbox.get(i) for i in self.home_train_user_listbox.curselection()] if self.home_train_user_listbox else []
        sel_labels = [self.home_train_label_listbox.get(i) for i in self.home_train_label_listbox.curselection()] if self.home_train_label_listbox else []
        if not sel_chs or not sel_users or not sel_labels:
            messagebox.showerror("Missing selection", "Select channels, users, and labels.")
            return
        model_type = self.training_model_type_var.get()
        self._set_training_report("Training in progress...")
        self.root.update_idletasks()
        try:
            result = train_named_model(mode, name, selected_users=sel_users, selected_labels=sel_labels,
                                       selected_channels=sel_chs, model_type=model_type)
        except Exception as exc:
            self._log(f"Training failed: {exc}", level="ERROR")
            self._set_training_report(f"Training failed:\n{exc}")
            messagebox.showerror("Training failed", str(exc))
            return
        bundle = result["bundle"]
        requested_type = bundle.get("requested_model_type_display", model_type)
        selected_type = bundle.get("model_type_display", model_type)
        benchmark_summary = bundle.get("benchmark_summary")
        type_lines = [f"Requested type: {requested_type}"]
        if requested_type != selected_type:
            type_lines.append(f"Selected type: {selected_type}")
        else:
            type_lines.append(f"Type: {selected_type}")
        type_summary = "\n".join(type_lines)
        benchmark_text = f"{benchmark_summary}\n\n" if benchmark_summary else ""
        heatmap_lines = []
        if bundle.get("confusion_heatmap_path"):
            heatmap_lines.append(f"Confusion heatmap: {bundle['confusion_heatmap_path']}")
        for item in bundle.get("benchmark_confusion_heatmap_paths") or []:
            heatmap_lines.append(
                f"  {item.get('model_type_display', item.get('model_type', '?'))}: {item.get('path')}"
            )
        heatmap_text = "\n".join(heatmap_lines)
        heatmap_block = f"{heatmap_text}\n" if heatmap_text else ""
        confusion_text = bundle.get("confusion_matrix_text")
        confusion_block = f"\n\n{confusion_text}" if confusion_text else ""
        benchmark_confusion_text = bundle.get("benchmark_confusion_matrices")
        benchmark_confusion_block = f"\n\n{benchmark_confusion_text}" if benchmark_confusion_text else ""
        self._set_training_report(
            f"BASE MODEL TRAINED\n"
            f"{'=' * 40}\n"
            f"Mode: {mode}\n"
            f"Model: {result['model_name']}\n"
            f"{type_summary}\n"
            f"Samples: {bundle['sample_count']}\n"
            f"Labels: {', '.join(bundle['labels'])}\n"
            f"{heatmap_block}\n"
            f"{benchmark_text}"
            f"{bundle['report']}"
            f"{confusion_block}"
            f"{benchmark_confusion_block}"
        )
        self._refresh_all_training()
        self.ft_base_var.set(result["model_name"])
        self.root.update_idletasks()
        self._log(
            f"Base model trained: {result['model_name']}, "
            f"type={selected_type}, requested={requested_type}, samples={bundle['sample_count']}"
        )
        messagebox.showinfo("Done", f"Base model saved: {result['model_name']}")

    def _refresh_finetune_lists(self):
        mode = self.training_mode_var.get()
        sel_chs = self._get_selected_train_channels() or []
        models = []
        for ch in (sel_chs or list(CHANNEL_NAMES)):
            for name in list_saved_models(mode, channel=ch):
                if name not in models:
                    models.append(name)
        self.ft_base_combo["values"] = models
        current = self.ft_base_var.get().strip()
        if current not in models:
            self.ft_base_var.set(models[0] if models else "")
        self._on_finetune_base_changed()

    def _on_finetune_base_changed(self):
        base_name = self.ft_base_var.get().strip()
        current_name = self.ft_new_model_name_var.get().strip()
        previous_default = f"{self._last_finetune_base_name}_continued" if self._last_finetune_base_name else ""
        if base_name and (not current_name or current_name == previous_default):
            self.ft_new_model_name_var.set(f"{base_name}_continued")
        self._last_finetune_base_name = base_name

    def _run_finetune(self):
        mode = self.training_mode_var.get()
        base_name = self.ft_base_var.get().strip()
        new_name = self.ft_new_model_name_var.get().strip()
        if not base_name:
            messagebox.showerror("Missing base model", "Select a saved base model first.")
            return
        if not new_name:
            messagebox.showerror("Missing name", "Enter a new model name.")
            return
        sel_chs = self._get_selected_train_channels() or []
        sel_users = [
            self.home_train_user_listbox.get(i)
            for i in self.home_train_user_listbox.curselection()
        ] if self.home_train_user_listbox else []
        if not sel_chs or not sel_users:
            messagebox.showerror("Missing selection", "Select channels and the users to add.")
            return

        self._set_training_report("Continuing training in progress...")
        self.root.update_idletasks()
        try:
            result = continue_training_from_base(
                mode,
                base_name,
                new_name,
                additional_users=sel_users,
                selected_channels=sel_chs,
            )
        except Exception as exc:
            self._log(f"Continue training failed: {exc}", level="ERROR")
            self._set_training_report(f"Continue training failed:\n{exc}")
            messagebox.showerror("Continue training failed", str(exc))
            return

        bundle = result["bundle"]
        heatmap_lines = []
        if bundle.get("confusion_heatmap_path"):
            heatmap_lines.append(f"Confusion heatmap: {bundle['confusion_heatmap_path']}")
        heatmap_text = "\n".join(heatmap_lines)
        heatmap_block = f"{heatmap_text}\n" if heatmap_text else ""
        confusion_text = bundle.get("confusion_matrix_text")
        confusion_block = f"\n\n{confusion_text}" if confusion_text else ""
        self._set_training_report(
            f"CONTINUED MODEL TRAINED\n"
            f"{'=' * 40}\n"
            f"Mode: {mode}\n"
            f"Base model: {bundle.get('continued_from_model', base_name)}\n"
            f"New model: {result['model_name']}\n"
            f"Type: {bundle.get('model_type_display', bundle.get('model_type', '?'))}\n"
            f"Base users: {', '.join(bundle.get('continued_from_users') or [])}\n"
            f"Added users: {', '.join(bundle.get('continued_with_users') or [])}\n"
            f"Samples: {bundle['sample_count']}\n"
            f"Labels: {', '.join(bundle['labels'])}\n"
            f"{heatmap_block}\n"
            f"{bundle['report']}"
            f"{confusion_block}"
        )
        self._refresh_all_training()
        self.ft_base_var.set(result["model_name"])
        self.ft_new_model_name_var.set(f"{result['model_name']}_continued")
        self.root.update_idletasks()
        self._log(
            f"Continued model trained: base={base_name}, new={result['model_name']}, "
            f"added_users={bundle.get('continued_with_users')}, samples={bundle['sample_count']}"
        )
        messagebox.showinfo("Done", f"Continued model saved: {result['model_name']}")

    def _delete_training_model(self):
        """Delete the model currently selected in the saved-model combo."""
        name = self.ft_base_var.get().strip()
        if not name:
            messagebox.showinfo("Select model", "Select a model in the 'Saved model' dropdown first.")
            return
        mode = self.training_mode_var.get()
        if not messagebox.askyesno("Confirm delete",
                                    f"Delete model '{name}' ({mode})?\n\n"
                                    "This will remove the .joblib and confusion matrix file(s) from disk."):
            return
        deleted = delete_named_model(mode, name)
        self._log(f"Deleted model '{name}' ({mode}): {deleted} file(s) removed.")
        self._refresh_all_training()
        messagebox.showinfo("Deleted", f"Model '{name}' deleted ({deleted} file(s)).")

    # ---------------------------------------------------------------
    # Testing
    # ---------------------------------------------------------------
    def _refresh_test_models(self):
        mode = self.testing_mode_var.get()
        # List models across ALL channel combinations
        models = list_saved_models(mode, selected_channels=list(CHANNEL_NAMES))
        if self.model_combo:
            self.model_combo["values"] = models
        if models:
            self.model_choice_var.set(models[0])
        else:
            self.model_choice_var.set("")
        self.test_model_info_var.set("")

    def _run_test(self):
        mode = self.testing_mode_var.get()
        name = self.model_choice_var.get().strip()
        if not name:
            messagebox.showerror("No model", "Select a model first.")
            return
        # Try loading from all possible channel directories
        bundle = None
        for ch_combo in [list(CHANNEL_NAMES), [CHANNEL_NAMES[0]], [CHANNEL_NAMES[1]]]:
            try:
                bundle = load_named_model(mode, name, selected_channels=ch_combo)
                if bundle is not None:
                    break
            except Exception:
                continue
        if bundle is None:
            messagebox.showerror("Load failed", f"Model '{name}' not found.")
            return
        bundle = dict(bundle)
        channels = bundle.get("selected_channels", [DEFAULT_CHANNEL])
        ch_display = ", ".join(channel_display_name(c) for c in channels)
        config = bundle.get("config", {})
        live_window = max(
            float(config.get("analysis_window_seconds", 0.0) or 0.0),
            float(config.get("buffer_seconds", 0.0) or 0.0),
        ) or None
        quiet_seconds = config.get("event_quiet_seconds")
        info = f"Channels: {ch_display}  |  Labels: {', '.join(bundle.get('labels', []))}"
        if mode == "single" and live_window is not None:
            info += f"  |  Live window: {live_window:.1f}s"
            if quiet_seconds is not None:
                info += f"  |  Quiet tail: {float(quiet_seconds):.2f}s"
        self.test_model_info_var.set(info)
        # Load calibration for the current user
        user = self.user_name_var.get().strip()
        calibration = load_calibration(user) if user else None
        self.calibration_data = calibration
        self._update_calibration_display()
        threshold = self.confidence_threshold_var.get()
        try:
            self.predictor = LivePredictor(bundle, calibration=calibration, display_threshold=threshold)
        except Exception as exc:
            self._log(f"Live test failed: {exc}", level="ERROR")
            messagebox.showerror("Live test failed", str(exc))
            return
        self.active_model_name = name
        self.last_predict_wall = 0.0
        self.prediction_hold_until = 0.0
        self.prediction_hold_label = None
        self.last_live_prediction_display = "N/A"
        self._clear_prediction_history()
        # Clear any previous debug dumps so a fresh test session starts empty.
        self.last_debug_reports.clear()
        self.debug_autocapture_wall = 0.0
        self.status_var.set("Testing")
        self.prediction_display_var.set("Waiting...")
        cal_info = "calibrated" if calibration else "no calibration"
        debug_info = " (Debug Mode ON)" if self.debug_mode_var.get() else ""
        self._log(f"Live test started: model={name}, mode={mode}, channels={channels}, {cal_info}, threshold={threshold:.2f}{debug_info}")

    def _on_debug_mode_toggle(self):
        """Called when the 'Debug Mode' checkbox is flipped. Resets the ring
        buffer so each debug session starts clean, and resets the throttle so
        the next auto-capture happens immediately."""
        if self.debug_mode_var.get():
            self.last_debug_reports.clear()
            self.debug_autocapture_wall = 0.0
            self._log("Debug Mode enabled — auto-capturing debug dumps. Click Export Debug Report when done.")
        else:
            self._log(f"Debug Mode disabled. {len(self.last_debug_reports)} dumps captured this session.")

    def _capture_debug_dump(self, log_to_panel=False):
        """Run one predictor.debug_dump() on the current live buffer and append
        the formatted text to self.last_debug_reports. Does NOT mutate the
        predictor. Used by Debug Mode auto-capture; optionally logs to the log
        panel so the user sees it on screen."""
        if self.predictor is None:
            return False
        channels = self.predictor.input_channels()
        buffers = {}
        for ch in channels:
            buf = self.channel_buffers.get(ch)
            if buf is None or not buf["time"]:
                return False
            buffers[ch] = {"time": list(buf["time"]), "voltage": list(buf["voltage"])}

        try:
            report = self.predictor.debug_dump(buffers)
        except Exception as exc:
            if log_to_panel:
                self._log(f"Debug capture failed: {exc}", level="ERROR")
            return False

        lines = [
            "===== PREDICTOR DEBUG DUMP =====",
            f"input_channels         : {report.get('input_channels')}",
            f"feature_names_count    : {report.get('feature_names_count')}",
            f"event_k                : {report.get('event_k')}",
            f"analysis_window_seconds: {report.get('analysis_window_seconds')}",
        ]

        if "error" in report:
            lines.append(f"ERROR                  : {report['error']}")
            err_text = "\n".join(lines)
            if log_to_panel:
                self._log(err_text)
            self.last_debug_reports.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "text": err_text,
                "raw": report,
            })
            return True

        lines += [
            f"aligned_sample_count   : {report.get('aligned_sample_count')}",
            f"aligned_duration_s     : {report.get('aligned_duration_seconds'):.3f}",
            f"min_points_to_predict  : {report.get('min_points_to_predict')}",
            f"fs_estimated_hz        : {report.get('fs_estimated_hz'):.1f}",
            f"fallback_threshold     : {report.get('fallback_threshold'):.4f}",
            "",
            "----- per channel -----",
        ]

        for ch, info in (report.get("channels") or {}).items():
            lines += [
                f"[{ch}]",
                f"  calibration_loaded   : {info['calibration_loaded']}",
                f"  calibration_sane     : {info['calibration_sane']}",
                f"  cal_voltage_mean     : {info['cal_voltage_mean']}",
                f"  cal_voltage_std      : {info['cal_voltage_std']}",
                f"  baseline_used        : {info['baseline_used']:.4f}",
                f"  noise_std_used       : {info['noise_std_used']}",
                f"  threshold            : {info['threshold']:.4f}",
                f"  raw signal           : min={info['raw_signal_min']:.3f}  max={info['raw_signal_max']:.3f}  mean={info['raw_signal_mean']:.3f}",
                f"  centered             : min={info['centered_min']:.3f}  max={info['centered_max']:.3f}  absmax={info['centered_absmax']:.3f}",
                f"  envelope             : min={info['envelope_min']:.4f}  max={info['envelope_max']:.4f}  mean={info['envelope_mean']:.4f}",
                f"  mask_true_count      : {info['mask_true_count']} / {report.get('aligned_sample_count')}",
                f"  mask_true_fraction   : {info['mask_true_fraction']:.2f}",
                f"  mask_last_sample_ON  : {info['mask_last_sample_active']}",
                f"  num_runs             : {info['num_runs']}",
                f"  longest_run          : {info['longest_run']} samples",
                f"  first 5 runs (start,len): {info['first_5_runs_start_len']}",
            ]

        lines += [
            "",
            "----- combined mask / event slice -----",
            f"min_samples_required   : {report.get('min_samples_required')}  (= {report.get('min_samples_required', 0) / max(report.get('fs_estimated_hz', 1), 1):.3f}s at current fs)",
            f"pad_samples            : {report.get('pad_samples')}",
            f"combined_mask_true_cnt : {report.get('combined_mask_true_count')}",
            f"combined_mask_true_frac: {report.get('combined_mask_true_fraction'):.2f}",
            f"combined_last_on       : {report.get('combined_mask_last_sample_active')}",
            f"tail_active_fraction   : {report.get('tail_active_fraction'):.2f}",
        ]

        if report.get("event_slice") is None and "event_slice_start" not in report:
            lines += [
                "event_slice            : NONE",
                f"would_display          : {report.get('would_display')}",
            ]
        else:
            lines += [
                f"event_slice            : [{report['event_slice_start']}, {report['event_slice_stop']})  length={report['event_slice_length']} samples  duration={report['event_slice_duration_seconds']:.3f}s",
                f"event_end_time         : {report['event_end_time']:.3f}",
                f"lag_seconds            : {report['lag_seconds']:.3f}  (max={report['max_completion_lag']:.3f})",
            ]
            if "classifier_prediction" in report:
                lines += [
                    f"classifier_prediction  : {report['classifier_prediction']}",
                    f"classifier_confidence  : {report['classifier_confidence']:.3f}",
                    f"confidence_threshold   : {report.get('confidence_threshold'):.2f}",
                    f"display_threshold      : {report.get('display_threshold'):.2f}",
                ]
            if "classifier_error" in report:
                lines.append(f"classifier_error       : {report['classifier_error']}")
            lines.append(f"would_display          : {report.get('would_display')}")

        lines.append("================================")
        full_text = "\n".join(lines)
        if log_to_panel:
            self._log(full_text)
        self.last_debug_reports.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "text": full_text,
            "raw": report,
        })
        return True

    def _export_debug_report(self):
        """Dump everything relevant for debugging (model bundle, training
        report, calibration, live buffer stats, current predictor state,
        recent debug dumps, recent predictions, recent logs) to a single
        .txt file. Intended for the user to send the file directly to me
        when something is wrong with training or live prediction."""
        from tkinter import filedialog
        import sys
        import platform

        default_name = f"emg_debug_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save debug report",
        )
        if not path:
            return

        lines = []

        def add(msg=""):
            lines.append(msg)

        def section(title):
            add("")
            add("=" * 70)
            add(title)
            add("=" * 70)

        section("EMG APP DEBUG REPORT")
        add(f"Generated        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        add(f"Python           : {sys.version.split()[0]}")
        add(f"Platform         : {platform.platform()}")
        try:
            import numpy as _np
            import sklearn as _sk
            import pandas as _pd
            add(f"numpy            : {_np.__version__}")
            add(f"scikit-learn     : {_sk.__version__}")
            add(f"pandas           : {_pd.__version__}")
        except Exception:
            pass
        try:
            import joblib as _jl
            add(f"joblib           : {_jl.__version__}")
        except Exception:
            pass

        section("Session")
        session_user = self.session_user or self.user_name_var.get().strip() or "(unknown)"
        add(f"User             : {session_user}")
        add(f"Active model     : {self.active_model_name or '(none)'}")
        add(f"Session mode     : {self.session_mode or '(idle)'}")
        add(f"Display threshold: {self.confidence_threshold_var.get():.2f}")

        section("Calibration")
        cal = self.calibration_data
        if cal is None:
            add("(not calibrated)")
        else:
            add(f"User             : {cal.get('user', '?')}")
            add(f"Timestamp        : {cal.get('timestamp', '?')}")
            add(f"Sample count     : {cal.get('sample_count', '?')}")
            add(f"event_threshold_k: {cal.get('event_threshold_k', '?')}")
            for ch, d in (cal.get("channels") or {}).items():
                std = float(d.get("voltage_std", 0.0) or 0.0)
                mean = float(d.get("voltage_mean", 0.0) or 0.0)
                flag = " (BROKEN)" if std > 0.05 else ""
                add(f"  [{ch}] mean={mean:.4f} V  std={std:.4f} V{flag}")

        section("Model Bundle")
        if self.predictor is None:
            add("(no predictor active — click Run Test first)")
        else:
            bundle = self.predictor.bundle
            add(f"Model name       : {bundle.get('model_name', '?')}")
            add(f"Mode             : {bundle.get('mode', '?')}")
            add(f"Model type       : {bundle.get('model_type_display') or bundle.get('model_type', '?')}")
            add(f"Requested type   : {bundle.get('requested_model_type_display') or bundle.get('requested_model_type', '?')}")
            add(f"Labels           : {bundle.get('labels')}")
            add(f"Selected channels: {bundle.get('selected_channels')}")
            add(f"Preprocessing ver: {bundle.get('preprocessing_version')}")
            add(f"Sample count     : {bundle.get('sample_count')}")
            add(f"Dataset source   : {bundle.get('dataset_dir', '?')}")
            add(f"Created at       : {bundle.get('created_at', '?')}")
            add(f"Confusion heatmap: {bundle.get('confusion_heatmap_path', '(none)')}")
            if bundle.get("continued_from_model"):
                add(f"Continued from   : {bundle.get('continued_from_model')}")
                add(f"Base users       : {bundle.get('continued_from_users')}")
                add(f"Added users      : {bundle.get('continued_with_users')}")
            feature_names = list(bundle.get("feature_names") or [])
            add(f"Feature count    : {len(feature_names)}")
            if feature_names:
                add(f"Feature names    : {feature_names}")

        if self.predictor is not None:
            section("Training Report (from bundle)")
            report_text = self.predictor.bundle.get("report") or "(no report stored in bundle)"
            add(report_text)

            confusion_text = self.predictor.bundle.get("confusion_matrix_text")
            if confusion_text:
                section("Training Confusion Matrix")
                add(confusion_text)

            bench = self.predictor.bundle.get("benchmark_summary")
            if bench:
                section("Auto Best Benchmark Summary")
                add(bench)

            benchmark_confusion_text = self.predictor.bundle.get("benchmark_confusion_matrices")
            if benchmark_confusion_text:
                section("Auto Best Confusion Matrices")
                add(benchmark_confusion_text)

            benchmark_heatmap_paths = self.predictor.bundle.get("benchmark_confusion_heatmap_paths")
            if benchmark_heatmap_paths:
                section("Auto Best Heatmap Files")
                for item in benchmark_heatmap_paths:
                    add(f"  {item.get('model_type_display', item.get('model_type', '?'))}: {item.get('path')}")

            bench_results = self.predictor.bundle.get("benchmark_results")
            if bench_results:
                section("Per-Model Benchmark Details")
                for r in bench_results:
                    add(
                        f"  {r.get('model_type_display', '?'):<22} "
                        f"macro-F1={r.get('macro_f1_mean', 0):.3f}  "
                        f"acc={r.get('accuracy_mean', 0):.3f}  "
                        f"splits={r.get('split_count', 0)}"
                    )

            section("Bundle Config")
            for k, v in (self.predictor.bundle.get("config") or {}).items():
                add(f"  {k}: {v}")

        section("Live Buffer (current)")
        for ch in CHANNEL_NAMES:
            buf = self.channel_buffers.get(ch, {})
            n = len(buf.get("voltage", []))
            if n == 0:
                add(f"  [{ch}] no data")
                continue
            v = np.asarray(list(buf["voltage"]), dtype=np.float64)
            t = list(buf.get("time", []))
            duration = (t[-1] - t[0]) if len(t) >= 2 else 0.0
            add(
                f"  [{ch}] n={n}  duration={duration:.2f}s  "
                f"voltage=[{float(v.min()):.3f}, {float(v.max()):.3f}]V  "
                f"mean={float(v.mean()):.4f}V  std={float(v.std()):.4f}V"
            )

        # Take a fresh debug dump right now if the predictor is live
        if self.predictor is not None:
            section("Fresh Predictor Debug Dump (taken now)")
            try:
                channels = self.predictor.input_channels()
                bufs = {}
                for ch in channels:
                    b = self.channel_buffers.get(ch)
                    if b is None or not b["time"]:
                        continue
                    bufs[ch] = {"time": list(b["time"]), "voltage": list(b["voltage"])}
                if not bufs:
                    add("(no live buffer data for the predictor's input channels)")
                else:
                    fresh = self.predictor.debug_dump(bufs)
                    # Reuse the compact formatter by temporarily printing into lines
                    add(f"input_channels         : {fresh.get('input_channels')}")
                    add(f"feature_names_count    : {fresh.get('feature_names_count')}")
                    add(f"event_k                : {fresh.get('event_k')}")
                    add(f"analysis_window_seconds: {fresh.get('analysis_window_seconds')}")
                    if "error" in fresh:
                        add(f"ERROR                  : {fresh['error']}")
                    else:
                        add(f"aligned_sample_count   : {fresh.get('aligned_sample_count')}")
                        add(f"aligned_duration_s     : {fresh.get('aligned_duration_seconds', 0):.3f}")
                        add(f"fs_estimated_hz        : {fresh.get('fs_estimated_hz', 0):.1f}")
                        add(f"fallback_threshold     : {fresh.get('fallback_threshold', 0):.4f}")
                        for ch, info in (fresh.get("channels") or {}).items():
                            add(f"[{ch}]")
                            add(f"  calibration_sane     : {info.get('calibration_sane')}")
                            add(f"  cal_voltage_mean     : {info.get('cal_voltage_mean')}")
                            add(f"  cal_voltage_std      : {info.get('cal_voltage_std')}")
                            add(f"  baseline_used        : {info.get('baseline_used'):.4f}")
                            add(f"  noise_std_used       : {info.get('noise_std_used')}")
                            add(f"  threshold            : {info.get('threshold'):.4f}")
                            add(f"  raw signal           : min={info.get('raw_signal_min'):.3f}  max={info.get('raw_signal_max'):.3f}  mean={info.get('raw_signal_mean'):.3f}")
                            add(f"  centered             : min={info.get('centered_min'):.3f}  max={info.get('centered_max'):.3f}  absmax={info.get('centered_absmax'):.3f}")
                            add(f"  envelope             : min={info.get('envelope_min'):.4f}  max={info.get('envelope_max'):.4f}  mean={info.get('envelope_mean'):.4f}")
                            add(f"  mask_true_count      : {info.get('mask_true_count')}/{fresh.get('aligned_sample_count')}")
                            add(f"  mask_true_fraction   : {info.get('mask_true_fraction'):.2f}")
                            add(f"  mask_last_sample_ON  : {info.get('mask_last_sample_active')}")
                            add(f"  num_runs             : {info.get('num_runs')}")
                            add(f"  longest_run          : {info.get('longest_run')}")
                            add(f"  first 5 runs         : {info.get('first_5_runs_start_len')}")
                        add(f"min_samples_required   : {fresh.get('min_samples_required')}")
                        add(f"combined_mask_true_cnt : {fresh.get('combined_mask_true_count')}")
                        add(f"combined_mask_true_frac: {fresh.get('combined_mask_true_fraction'):.2f}")
                        add(f"tail_active_fraction   : {fresh.get('tail_active_fraction'):.2f}")
                        if "event_slice_start" in fresh:
                            add(
                                f"event_slice            : [{fresh['event_slice_start']}, {fresh['event_slice_stop']})  "
                                f"length={fresh['event_slice_length']} samples  "
                                f"duration={fresh['event_slice_duration_seconds']:.3f}s"
                            )
                            add(f"lag_seconds            : {fresh.get('lag_seconds', 0):.3f}  (max={fresh.get('max_completion_lag', 0):.3f})")
                            if "classifier_prediction" in fresh:
                                add(f"classifier_prediction  : {fresh['classifier_prediction']}")
                                add(f"classifier_confidence  : {fresh['classifier_confidence']:.3f}")
                        add(f"would_display          : {fresh.get('would_display')}")
            except Exception as exc:
                add(f"(fresh debug dump failed: {exc})")

        if self.last_debug_reports:
            section(f"Historical Debug Dumps ({len(self.last_debug_reports)} captured)")
            for i, entry in enumerate(self.last_debug_reports):
                add(f"--- [{i + 1}] {entry['timestamp']} ---")
                add(entry["text"])
                add("")

        # Prediction history from the tree widget
        if self.prediction_history_tree is not None:
            section("Recent Predictions (from GUI history)")
            items = self.prediction_history_tree.get_children()
            if not items:
                add("(no predictions in history yet)")
            else:
                add("time      gesture          confidence")
                add("-" * 42)
                for item in items[:80]:
                    values = self.prediction_history_tree.item(item, "values")
                    if len(values) >= 3:
                        add(f"{values[0]:<10}{str(values[1]):<17}{values[2]}")

        # Log panel tail
        try:
            log_widget = self.log_widgets.get("testing") or next(iter(self.log_widgets.values()))
        except Exception:
            log_widget = None
        if log_widget is not None:
            try:
                log_text = log_widget.get("1.0", "end")
            except Exception:
                log_text = ""
            if log_text.strip():
                section("Log Panel Tail (last ~200 lines)")
                tail = log_text.splitlines()[-200:]
                for raw_line in tail:
                    add(raw_line)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._log(f"Debug report saved: {path}")
            messagebox.showinfo(
                "Debug Report Saved",
                f"Saved to:\n{path}\n\nYou can send this file directly to paste into chat.",
            )
        except Exception as exc:
            self._log(f"Debug report save failed: {exc}", level="ERROR")
            messagebox.showerror("Save failed", str(exc))

    def _stop_predictor(self):
        self.predictor = None
        self.active_model_name = None
        self.prediction_hold_until = 0.0
        self.prediction_hold_label = None
        self.last_live_prediction_display = "N/A"
        self.prediction_var.set("Prediction: --")
        self.prediction_display_var.set("--")

    def _clear_prediction_history(self):
        if self.prediction_history_tree is None:
            return
        self.prediction_history_tree.delete(*self.prediction_history_tree.get_children())

    def _append_prediction_history(self, gesture, confidence):
        if self.prediction_history_tree is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        self.prediction_history_tree.insert(
            "",
            0,
            values=(timestamp, gesture, f"{confidence:.2f}"),
        )
        children = self.prediction_history_tree.get_children()
        if len(children) > PREDICTION_HISTORY_LIMIT:
            self.prediction_history_tree.delete(*children[PREDICTION_HISTORY_LIMIT:])

    # ---------------------------------------------------------------
    # Calibration & confidence
    # ---------------------------------------------------------------
    def _run_calibration(self):
        """Capture current idle data from serial buffers and compute noise floor."""
        if self.ser is None or time.time() < self.serial_ready_at:
            messagebox.showerror("Not ready", "Connect to Arduino first and wait for warmup.")
            return

        channels = self._current_channels()
        # Pull the ENTIRE rolling channel buffer (up to PLOT_WINDOW_SECONDS of
        # data). Then for each channel, find the quietest 1-second sub-window
        # inside that buffer — that's our best estimate of "actual rest".
        # Averaging the whole buffer (the old behavior) silently includes any
        # gesture activity and produces a broken calibration with voltage_std
        # inflated by 1000x, which locks the predictor in 'capturing' forever.
        channels_data = {}
        min_samples = None
        quietest_window_seconds = 1.0
        SANE_STD_CEILING = 0.05  # must stay in sync with emg_model_tools.MAX_REASONABLE_NOISE_STD

        for ch in channels:
            buf = self.channel_buffers[ch]
            if len(buf["voltage"]) < 100:
                messagebox.showerror("Not enough data",
                                     f"Channel {ch} has too few samples ({len(buf['voltage'])}).\n"
                                     "Keep your hand still and wait a few seconds, then try again.")
                return
            voltages = np.asarray(list(buf["voltage"]), dtype=np.float32)
            adcs = np.asarray(list(buf["adc"]), dtype=np.float32)
            times = np.asarray(list(buf["time"]), dtype=np.float32)

            # Estimate sampling rate from the time buffer (fs can vary from
            # 50-500 Hz across hardware setups).
            if len(times) >= 2:
                dt = float(np.mean(np.diff(times)))
                fs_est = 1.0 / dt if dt > 0 else 200.0
            else:
                fs_est = 200.0
            win_len = max(10, int(round(fs_est * quietest_window_seconds)))
            win_len = min(win_len, len(voltages))

            if win_len >= len(voltages):
                quiet_voltages = voltages
                quiet_adcs = adcs
            else:
                # Sliding window; pick the window with minimum std
                best_start = 0
                best_std = float("inf")
                step = max(1, win_len // 10)
                for start in range(0, len(voltages) - win_len + 1, step):
                    window = voltages[start:start + win_len]
                    s = float(np.std(window))
                    if s < best_std:
                        best_std = s
                        best_start = start
                quiet_voltages = voltages[best_start:best_start + win_len]
                quiet_adcs = adcs[best_start:best_start + win_len]

            channels_data[ch] = {
                "voltage_mean": float(np.mean(quiet_voltages)),
                "voltage_std": float(np.std(quiet_voltages)),
                "adc_mean": float(np.mean(quiet_adcs)),
                "adc_std": float(np.std(quiet_adcs)),
            }
            n = len(quiet_voltages)
            if min_samples is None or n < min_samples:
                min_samples = n

        # Sanity check: if std is still huge after picking the quietest
        # sub-window, the user is not at rest at all. Refuse to save.
        bad_channels = [
            ch for ch, d in channels_data.items()
            if d["voltage_std"] > SANE_STD_CEILING
        ]
        if bad_channels:
            bad_info = "\n".join(
                f"  {ch}: mean={channels_data[ch]['voltage_mean']:.3f}V  std={channels_data[ch]['voltage_std']:.4f}V"
                for ch in bad_channels
            )
            messagebox.showerror(
                "Calibration looks wrong",
                "Even the quietest 1-second window has way too much variation "
                f"on these channels:\n\n{bad_info}\n\n"
                "Keep your hand COMPLETELY relaxed (no gestures, no tensing) "
                "and click Calibrate Idle again. The existing calibration was "
                "NOT saved."
            )
            return

        user = self.session_user or self.user_name_var.get().strip()
        if not user:
            messagebox.showerror("No user", "Enter a user name first.")
            return

        calibration_data = {
            "user": user,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "channels": channels_data,
            "sample_count": min_samples,
            "event_threshold_k": 3.0,
        }

        save_calibration(user, calibration_data)
        self.calibration_data = calibration_data
        self._update_calibration_display()

        # Update live predictor if running
        if self.predictor is not None:
            self.predictor.set_calibration(calibration_data)

        ch_info = "  ".join(
            f"{ch}: mean={d['voltage_mean']:.3f}V std={d['voltage_std']:.4f}V"
            for ch, d in channels_data.items()
        )
        self._log(f"Calibration saved for {user}: {ch_info}")
        messagebox.showinfo("Calibration",
                            f"Noise floor calibrated for {user}.\n\n" +
                            "\n".join(f"{ch}: {d['voltage_mean']:.3f} +/- {d['voltage_std']:.4f} V"
                                      for ch, d in channels_data.items()))

    def _update_calibration_display(self):
        """Update the calibration status label."""
        cal = self.calibration_data
        if cal is None:
            self.calibration_status_var.set("Not calibrated")
            return
        channels = cal.get("channels", {})
        ch_parts = []
        broken = []
        SANE_STD_CEILING = 0.05
        for ch, d in channels.items():
            std = float(d.get("voltage_std", 0.0) or 0.0)
            if std > SANE_STD_CEILING:
                broken.append(ch)
                ch_parts.append(f"{ch}: BAD ({std:.2f}V std)")
            else:
                ch_parts.append(f"{ch}: {d['voltage_mean']:.2f}V")
        prefix = "Cal (BROKEN — recalibrate!): " if broken else "Cal: "
        self.calibration_status_var.set(prefix + ", ".join(ch_parts))

    def _on_threshold_change(self, *_args):
        """Called when the confidence threshold slider is moved."""
        val = self.confidence_threshold_var.get()
        self.threshold_display_var.set(f"{val:.2f}")
        if self.predictor is not None:
            self.predictor.set_display_threshold(val)

    # ---------------------------------------------------------------
    # Capture history (right panel)
    # ---------------------------------------------------------------
    def _refresh_capture_list(self):
        if self.capture_tree is None:
            return
        self.capture_tree.delete(*self.capture_tree.get_children())
        if not self.session_user:
            self.data_summary_var.set("No user selected.")
            return
        records = list_user_capture_records(self.session_user)
        if not records:
            self.data_summary_var.set(f"No data for {self.session_user}.")
            return

        # aggregate counts by label+mode
        counts = {}
        for rec in records:
            key = (rec["label"], rec["mode"], ", ".join(channel_display_name(ch) for ch in rec.get("channels", [])))
            counts[key] = counts.get(key, 0) + 1

        for (label, mode, ch_text), cnt in sorted(counts.items()):
            self.capture_tree.insert("", "end", values=(label, cnt, mode, ch_text))

        self.data_summary_var.set(f"{len(records)} captures for {self.session_user}")

    def _clear_user_data(self):
        if self.prepare_state or self.active_capture or self.auto_running:
            messagebox.showerror("Busy", "Stop the current capture first.")
            return
        raw = (self.session_user or self.user_name_var.get()).strip()
        if not raw:
            return
        safe = sanitize_user_name(raw)
        if not messagebox.askyesno("Confirm", f"Delete all data for '{safe}'?"):
            return
        result = clear_user_data(safe)
        self._log(f"Cleared user data: {safe}, deleted {result['deleted_files']} files.")
        self._refresh_capture_list()

    # ---------------------------------------------------------------
    # Serial polling & plot
    # ---------------------------------------------------------------
    def poll_serial(self):
        try:
            now = time.time()
            if self.ser is not None:
                self._read_serial()
                self._advance_capture()
                self._advance_prediction()
                if self._plot_dirty and now - self._last_plot_refresh_at >= (PLOT_REFRESH_MS / 1000.0):
                    self._refresh_plot()
                    self._last_plot_refresh_at = now
                    self._plot_dirty = False
                if now - self._last_banner_refresh_at >= (BANNER_REFRESH_MS / 1000.0):
                    self._refresh_banner()
                    self._last_banner_refresh_at = now
        except (SERIAL_EXCEPTION, OSError) as exc:
            self._log(f"Serial error: {exc}", level="ERROR")
            self._close_serial()
            self._reset_capture_state()
            self._reset_guided_state()
            self.status_var.set("Disconnected")
            self.banner_var.set("DEVICE DISCONNECTED")
            self.banner_timer_var.set("")
            self.banner_detail_var.set(str(exc))
            messagebox.showerror("Serial error", f"Device disconnected:\n{exc}")
        self.root.after(SERIAL_POLL_MS, self.poll_serial)

    def _read_serial(self):
        got_samples = False
        while self.ser.in_waiting:
            raw = self.ser.readline().decode("utf-8", errors="ignore")
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                if stripped and "a1" in stripped.lower():
                    self.detected_channels.update(CHANNEL_NAMES)
                continue
            result = parse_line(raw)
            if result is None:
                continue
            parsed, _ = result
            self.detected_channels.update(parsed.keys())
            for ch, (t, adc, voltage) in parsed.items():
                buf = self.channel_buffers[ch]
                if buf["time"] and t < (buf["time"][-1] - 1e-6):
                    self._log(f"Time reset detected on {ch}; clearing live buffers.", level="INFO")
                    self._reset_plot_buffers()
                    if self.predictor is not None:
                        self.predictor.reset()
                    buf = self.channel_buffers[ch]
                buf["time"].append(t)
                buf["adc"].append(adc)
                buf["voltage"].append(voltage)
                got_samples = True
                if (time.time() >= self.serial_ready_at
                        and self.active_capture is not None
                        and ch in self.active_capture["rows_by_channel"]):
                    elapsed = time.time() - self.active_capture["started_at"]
                    self.active_capture["rows_by_channel"][ch].append(
                        [f"{elapsed:.4f}", adc, f"{voltage:.5f}", self.active_capture["label"]]
                    )
        self._trim_buffers()
        if got_samples:
            self._plot_dirty = True

    def _trim_buffers(self):
        for ch in CHANNEL_NAMES:
            buf = self.channel_buffers[ch]
            t_buf = buf["time"]
            if not t_buf:
                continue
            latest = t_buf[-1]
            while t_buf and (latest - t_buf[0] > PLOT_WINDOW_SECONDS):
                t_buf.popleft()
                buf["adc"].popleft()
                buf["voltage"].popleft()

    def _advance_capture(self):
        if self.ser is None or time.time() < self.serial_ready_at:
            return
        now = time.time()
        # break state waits for user action (Space key) — no timer check
        # --- prepare countdown ---
        if self.prepare_state is not None and now >= self.prepare_state["deadline"]:
            ps = self.prepare_state
            self._begin_capture(ps["label"], ps["duration"], ps["auto"], ps.get("capture_mode"), ps.get("guided", False), ps.get("no_save", False))
        # --- capture duration ---
        if self.active_capture is not None:
            elapsed = now - self.active_capture["started_at"]
            if elapsed >= self.active_capture["duration"]:
                self._finish_capture()

    def _advance_prediction(self):
        if self.predictor is None:
            return
        channels = self.predictor.input_channels()
        buffers = {}
        for ch in channels:
            buf = self.channel_buffers.get(ch)
            if buf is None or not buf["time"]:
                return
            buffers[ch] = {"time": buf["time"], "voltage": buf["voltage"]}
        now = time.time()
        if now - self.last_predict_wall < self.predictor.predict_every_seconds:
            return
        snap = self.predictor.predict(buffers)
        self.last_predict_wall = now
        display = snap["display_prediction"]
        raw = snap["raw_prediction"]
        conf = snap["confidence"]
        is_idle = snap.get("is_idle", False)
        tail_frac = snap.get("tail_active_fraction")
        transient_states = {"idle", "capturing", "unknown", "N/A", "--", "Waiting..."}
        holding_prediction = False

        if display not in transient_states:
            self.prediction_hold_label = display
            self.prediction_hold_until = now + PREDICTION_HOLD_SECONDS
            if self.last_live_prediction_display != display:
                self._append_prediction_history(display, conf)
        elif self.prediction_hold_label and now < self.prediction_hold_until:
            holding_prediction = True

        shown_display = self.prediction_hold_label if holding_prediction else display
        self.last_live_prediction_display = display

        tail_text = ""
        if tail_frac is not None:
            tail_text = f" | tail_active={tail_frac:.2f}"

        if is_idle:
            info = f"State: idle | Raw: {raw} | Conf: {conf:.2f}{tail_text}"
        else:
            info = (
                f"Display: {display} | Raw: {raw} | Conf: {conf:.2f}"
                f" | Thresh: {self.confidence_threshold_var.get():.2f}{tail_text}"
            )
        if holding_prediction:
            info = f"Recent: {shown_display} | {info}"

        self.prediction_var.set(info)
        self.prediction_display_var.set(shown_display)

        # Color the prediction label based on state
        if hasattr(self, "prediction_display_label"):
            if holding_prediction or shown_display not in transient_states:
                self.prediction_display_label.configure(foreground="#0f766e")  # teal
            elif is_idle:
                self.prediction_display_label.configure(foreground="#94a3b8")  # gray
            elif shown_display == "unknown":
                self.prediction_display_label.configure(foreground="#f59e0b")  # amber
            else:
                self.prediction_display_label.configure(foreground="#0f766e")  # teal

        # Debug Mode auto-capture: once per debug_autocapture_interval, record
        # a full debug dump into the ring buffer. This is what feeds Export
        # Debug Report when the user clicks it.
        if self.debug_mode_var.get() and (
            now - self.debug_autocapture_wall >= self.debug_autocapture_interval
        ):
            self.debug_autocapture_wall = now
            try:
                self._capture_debug_dump(log_to_panel=False)
            except Exception as exc:
                # Never let a debug-capture hiccup break the prediction loop.
                self._log(f"Debug auto-capture error: {exc}", level="ERROR")

    def _detect_rest(self):
        buf = self.channel_buffers[DEFAULT_CHANNEL]
        if len(buf["voltage"]) < 20:
            return False
        signal = np.asarray(list(buf["voltage"])[-100:], dtype=np.float32)
        return float(np.mean(np.abs(signal))) < REST_THRESHOLD

    def _refresh_plot(self):
        # choose which plot to update based on current page
        if self.session_mode == "test":
            ax = self.test_plot_axis
            lines = self.test_lines
            cv = self.test_canvas
        elif self.session_mode in ("single", "session"):
            ax = self.plot_axis
            lines = self.lines_by_channel
            cv = self.canvas
        else:
            return

        visible = self._current_channels()
        use_voltage = self.plot_unit_var.get() == "voltage"
        y_vals = []
        max_pts = 0

        for ch in CHANNEL_NAMES:
            line = lines[ch]
            buf = self.channel_buffers[ch]
            val_buf = buf["voltage"] if use_voltage else buf["adc"]
            if ch not in visible or len(val_buf) < 2:
                line.set_data([], [])
                continue
            y = np.asarray(val_buf, dtype=np.float32)
            x = np.arange(len(y), dtype=np.float32)
            line.set_data(x, y)
            y_vals.append(y)
            max_pts = max(max_pts, len(y))

        if max_pts > 1:
            ax.set_xlim(0, max_pts - 1)

        # Y-axis: start with a sensible default, only expand, never shrink
        if y_vals:
            stacked = np.concatenate(y_vals)
            data_min = float(np.min(stacked))
            data_max = float(np.max(stacked))
            margin = max(0.05 * (data_max - data_min), 0.2 if use_voltage else 10.0)
            want_lo = data_min - margin
            want_hi = data_max + margin
            cur_lo, cur_hi = ax.get_ylim()
            # only expand outward
            ax.set_ylim(min(cur_lo, want_lo), max(cur_hi, want_hi))

        label = self.selected_label_var.get() or "-"
        state = self.status_var.get()
        user = self.session_user or "-"
        ax.set_title(f"Live EMG | user={user} | label={label} | {state}")
        cv.draw_idle()

    def _guided_detail(self, prefix=""):
        """Build guided session progress string: Gesture X/Y | Collection X/Y | Next: ..."""
        if not self.guided_session_active:
            return prefix
        idx = self.guided_session_step_index
        tasks = self.guided_session_tasks
        if idx >= len(tasks):
            return prefix
        task = tasks[idx]
        gn = task.get("gesture_name", "?").upper()
        gm = task.get("gesture_mode", "")
        gi = task.get("gesture_index", 0)
        gt = task.get("gesture_total", 0)
        ci = task.get("collection_index", 0)
        ct = task.get("collection_total", 0)
        mode_label = "Single" if gm == "single" else "Continuous"

        parts = [prefix] if prefix else []
        parts.append(f"Gesture {gi}/{gt} ({gn} \u2014 {mode_label})")
        parts.append(f"Collection {ci}/{ct}")

        # Find next gesture group
        for t in tasks[idx:]:
            t_gi = t.get("gesture_index", 0)
            if t.get("capture_mode") != "break" and t_gi != gi:
                next_gn = t.get("gesture_name", "?").upper()
                next_gm = "Single" if t.get("gesture_mode") == "single" else "Continuous"
                parts.append(f"Next: {next_gn} ({next_gm})")
                break
        else:
            parts.append("Last gesture!")

        return "  |  ".join(parts)

    def _refresh_banner(self):
        if self.session_mode not in ("single", "session"):
            return
        now = time.time()
        if now < self.serial_ready_at:
            remaining = max(0, self.serial_ready_at - now)
            self.banner_var.set("WAITING FOR ARDUINO RESET")
            self.banner_timer_var.set(f"{remaining:.1f}s")
            self.banner_detail_var.set("")
            return

        # --- break between gesture groups (waits for user) ---
        if self.guided_break_state is not None:
            task = self.guided_break_state["task"]
            next_gn = task.get("next_gesture_name", "?").upper()
            next_gm = "Single"
            next_gi = task.get("next_gesture_index", 0)
            gt = task.get("gesture_total", 0)
            next_ct = task.get("next_collection_total", 0)
            self.banner_var.set("BREAK \u2014 Press Space to continue")
            self.banner_timer_var.set("")
            detail = f"Up next: Gesture {next_gi}/{gt} {next_gn} ({next_gm})"
            detail += f"  |  {next_ct} collections"
            self.banner_detail_var.set(detail)
            self._flash_banner_if_new(f"gi_{next_gi}")
            return

        # --- active capture ---
        if self.active_capture is not None:
            remaining = max(0, self.active_capture["duration"] - (now - self.active_capture["started_at"]))
            label = self.active_capture["label"]
            if label == "rest":
                self.banner_var.set("REST NOW")
            else:
                self.banner_var.set(f"DO {label.upper()} NOW")
            self.banner_timer_var.set(f"{remaining:.1f}s")
            if self.guided_session_active:
                self.banner_detail_var.set(self._guided_detail())
            else:
                self.banner_detail_var.set("")
            return

        # --- prepare countdown ---
        if self.prepare_state is not None:
            remaining = max(0, self.prepare_state["deadline"] - now)
            label = self.prepare_state["label"]
            if label == "rest":
                self.banner_var.set("REST NOW")
            else:
                self.banner_var.set(f"GET READY: {label.upper()}")
            self.banner_timer_var.set(f"{remaining:.1f}s")
            if self.guided_session_active:
                self.banner_detail_var.set(self._guided_detail())
                idx = self.guided_session_step_index
                tasks = self.guided_session_tasks
                if idx < len(tasks):
                    self._flash_banner_if_new(f"gi_{tasks[idx].get('gesture_index', 0)}")
            else:
                self._flash_banner_if_new(label)
                self.banner_detail_var.set("")
            return

        # --- waiting for user (guided manual) ---
        if self.guided_session_active and self.guided_session_step_index < len(self.guided_session_tasks):
            task = self.guided_session_tasks[self.guided_session_step_index]
            gn = task.get("gesture_name", task.get("label", "?"))
            self.banner_var.set(f"UP NEXT: {gn.upper()}")
            self.banner_timer_var.set("")
            detail = self._guided_detail("Press Space to start")
            self.banner_detail_var.set(detail)
            return

        # --- default idle ---
        label = self.selected_label_var.get()
        self.banner_var.set("READY \u2014 Select gesture and press Start")
        self.banner_timer_var.set("")
        self.banner_detail_var.set(f"Current: {label}")

    # ---------------------------------------------------------------
    # Banner flash on gesture switch
    # ---------------------------------------------------------------
    def _flash_banner_if_new(self, label):
        """Flash the banner background briefly when the gesture label changes."""
        if label == self._last_flash_label:
            return
        self._last_flash_label = label

        # Cancel any pending flash-back timer
        if self._flash_after_id is not None:
            self.root.after_cancel(self._flash_after_id)
            self._flash_after_id = None

        flash_bg = "#facc15"  # bright yellow
        normal_bg = "#0f172a"

        self.banner_frame_widget.config(bg=flash_bg)
        self.banner_top_frame.config(bg=flash_bg)
        self.banner_label.config(bg=flash_bg, fg="#0f172a")
        self.banner_timer_label.config(bg=flash_bg, fg="#0f172a")
        self.banner_spacer.config(bg=flash_bg)
        self.banner_detail_label.config(bg=flash_bg, fg="#0f172a")

        def _restore():
            self.banner_frame_widget.config(bg=normal_bg)
            self.banner_top_frame.config(bg=normal_bg)
            self.banner_label.config(bg=normal_bg, fg="#f8fafc")
            self.banner_timer_label.config(bg=normal_bg, fg="#fbbf24")
            self.banner_spacer.config(bg=normal_bg)
            self.banner_detail_label.config(bg=normal_bg, fg="#cbd5e1")
            self._flash_after_id = None

        self._flash_after_id = self.root.after(400, _restore)

    # ---------------------------------------------------------------
    # Key bindings
    # ---------------------------------------------------------------
    def _on_key_press(self, event):
        if self.session_mode not in ("single", "session"):
            return
        if self.session_mode == "session":
            if event.keysym == "space" and self.guided_break_state is not None:
                self._finish_break()
            return
        key = (event.char or "").lower()
        label_map = {"1": "fist", "2": "open", "3": "one", "r": "rest"}
        if key in label_map:
            self.selected_label_var.set(label_map[key])
        elif event.keysym == "space":
            self._start_single_manual()

    # ---------------------------------------------------------------
    # USER DATA PAGE (browse captures & plot EMG)
    # ---------------------------------------------------------------
    def _build_user_data_page(self):
        page = ttk.Frame(self.container)
        self.pages["user_data"] = page

        # top bar
        top = ttk.Frame(page, padding=(12, 8))
        top.pack(fill="x")
        ttk.Button(top, text="< Back", command=lambda: self._show_page("home")).pack(side="left")
        ttk.Label(top, text="User Data Browser", font=("Helvetica", 20, "bold")).pack(side="left", padx=20)

        # main body: left panel (users + entries) | right panel (plot)
        body = ttk.PanedWindow(page, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        # --- left: user list + capture entries ---
        left = ttk.Frame(body)
        body.add(left, weight=1)

        # user list
        ttk.Label(left, text="Users", font=("Helvetica", 13, "bold")).pack(anchor="w", padx=4, pady=(4, 2))
        user_list_frame = ttk.Frame(left)
        user_list_frame.pack(fill="x", padx=4, pady=(0, 8))
        self.ud_user_listbox = tk.Listbox(user_list_frame, font=("Helvetica", 12), height=6,
                                          exportselection=False)
        ud_user_scroll = ttk.Scrollbar(user_list_frame, orient="vertical", command=self.ud_user_listbox.yview)
        self.ud_user_listbox.configure(yscrollcommand=ud_user_scroll.set)
        self.ud_user_listbox.pack(side="left", fill="both", expand=True)
        ud_user_scroll.pack(side="right", fill="y")
        self.ud_user_listbox.bind("<<ListboxSelect>>", self._ud_on_user_select)

        # capture entries with mode filter
        cap_header = ttk.Frame(left)
        cap_header.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Label(cap_header, text="Captures", font=("Helvetica", 13, "bold")).pack(side="left")
        for text, val in [("All", "all"), ("Single", "single")]:
            ttk.Radiobutton(cap_header, text=text, value=val,
                            variable=self.ud_mode_filter_var,
                            command=self._ud_on_user_select).pack(side="left", padx=(8, 0))
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        cols = ("label", "mode", "channels", "timestamp")
        self.ud_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
        self.ud_tree.heading("label", text="Label")
        self.ud_tree.heading("mode", text="Mode")
        self.ud_tree.heading("channels", text="Channels")
        self.ud_tree.heading("timestamp", text="Timestamp")
        self.ud_tree.column("label", width=100)
        self.ud_tree.column("mode", width=80)
        self.ud_tree.column("channels", width=80)
        self.ud_tree.column("timestamp", width=140)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.ud_tree.yview)
        self.ud_tree.configure(yscrollcommand=tree_scroll.set)
        self.ud_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.ud_tree.bind("<<TreeviewSelect>>", self._ud_on_entry_select)

        self.ud_entry_count_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.ud_entry_count_var, font=("Helvetica", 11)).pack(anchor="w", padx=4)

        ud_btn_frame = ttk.Frame(left)
        ud_btn_frame.pack(fill="x", padx=4, pady=(6, 4))
        ttk.Button(ud_btn_frame, text="Delete Selected Capture",
                   command=self._ud_delete_capture).pack(side="left", padx=(0, 6))
        ttk.Button(ud_btn_frame, text="Delete All User Data",
                   command=self._ud_delete_user).pack(side="left")

        # store capture records for lookup when clicked
        self.ud_records = []

        # --- right: plot ---
        right = ttk.Frame(body)
        body.add(right, weight=2)

        self.ud_plot_info_var = tk.StringVar(value="Select a capture to view its EMG data")
        ttk.Label(right, textvariable=self.ud_plot_info_var, font=("Helvetica", 12)).pack(anchor="w", padx=8, pady=4)

        ud_controls = ttk.Frame(right)
        ud_controls.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(ud_controls, text="View").pack(side="left")
        for text, val in [("Raw", "raw"), ("Processed", "processed"), ("Overlay", "overlay")]:
            ttk.Radiobutton(
                ud_controls,
                text=text,
                value=val,
                variable=self.ud_plot_mode_var,
                command=self._ud_redraw_current_capture,
            ).pack(side="left", padx=(8, 0))
        ttk.Button(
            ud_controls,
            text="Export Processed CSV",
            command=self._ud_export_processed_capture,
        ).pack(side="right")
        self.ud_compare_btn = ttk.Button(
            ud_controls,
            text="Compare...",
            command=self._ud_toggle_compare,
        )
        self.ud_compare_btn.pack(side="right", padx=(0, 6))

        self.ud_process_summary = tk.Text(
            right,
            height=8,
            wrap="word",
            font=("Courier", 10),
            state="disabled",
            bg="#fafafa",
        )
        self.ud_process_summary.pack(fill="x", padx=8, pady=(0, 4))

        fig = Figure(figsize=(7, 4), dpi=100)
        fig.set_facecolor("#f0f0f0")
        self.ud_plot_axis = fig.add_subplot(111)
        self.ud_plot_axis.set_xlabel("Time (s)")
        self.ud_plot_axis.set_ylabel("Voltage (V)")
        self.ud_plot_axis.set_title("EMG Signal")
        self.ud_canvas = FigureCanvasTkAgg(fig, master=right)
        self.ud_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _go_user_data(self):
        self._ud_refresh_users()
        self._show_page("user_data")

    def _ud_refresh_users(self):
        users = sorted(list_available_users("single"))
        self.ud_user_listbox.delete(0, "end")
        for u in users:
            self.ud_user_listbox.insert("end", u)
        # clear entries and plot
        self.ud_tree.delete(*self.ud_tree.get_children())
        self.ud_records = []
        self.ud_selected_record = None
        self.ud_last_preview = None
        self.ud_entry_count_var.set(f"{len(users)} user(s)")
        self.ud_plot_axis.cla()
        self.ud_plot_axis.set_xlabel("Time (s)")
        self.ud_plot_axis.set_ylabel("Voltage (V)")
        self.ud_plot_axis.set_title("EMG Signal")
        self.ud_plot_info_var.set("Select a capture to view its EMG data")
        self._set_ud_process_summary("")
        self.ud_canvas.draw_idle()

    def _ud_on_user_select(self, _event=None):
        sel = self.ud_user_listbox.curselection()
        if not sel:
            return
        user = self.ud_user_listbox.get(sel[0])
        self.ud_selected_user = user
        all_records = list_user_capture_records(user)
        mode_filter = self.ud_mode_filter_var.get()
        if mode_filter != "all":
            records = [r for r in all_records if r.get("mode") == mode_filter]
        else:
            records = all_records
        self.ud_records = records
        self.ud_selected_record = None
        self.ud_last_preview = None
        self.ud_tree.delete(*self.ud_tree.get_children())
        for i, rec in enumerate(records):
            chs = ", ".join(channel_display_name(c) for c in rec.get("channels", []))
            self.ud_tree.insert("", "end", iid=str(i), values=(
                rec["label"], rec["mode"], chs, rec.get("timestamp", "")
            ))
        filter_text = f" ({mode_filter})" if mode_filter != "all" else ""
        self.ud_entry_count_var.set(f"{len(records)} capture(s) for {user}{filter_text}  (total: {len(all_records)})")
        self.ud_plot_axis.cla()
        self.ud_plot_axis.set_xlabel("Time (s)")
        self.ud_plot_axis.set_ylabel("Voltage (V)")
        self.ud_plot_axis.set_title("EMG Signal")
        self.ud_plot_info_var.set("Select a capture to view its EMG data")
        self._set_ud_process_summary("")
        self.ud_canvas.draw_idle()

    def _ud_on_entry_select(self, _event=None):
        sel = self.ud_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self.ud_records):
            return
        self.ud_selected_record = self.ud_records[idx]
        self._ud_render_selected_capture()

    def _set_ud_process_summary(self, text):
        self.ud_process_summary.configure(state="normal")
        self.ud_process_summary.delete("1.0", "end")
        if text:
            self.ud_process_summary.insert("1.0", text)
        self.ud_process_summary.configure(state="disabled")

    def _ud_render_selected_capture(self):
        rec = self.ud_selected_record
        if rec is None:
            return

        user_sel = self.ud_user_listbox.curselection()
        user = self.ud_user_listbox.get(user_sel[0]) if user_sel else ""
        calibration = load_calibration(user) if user else None

        try:
            preview = prepare_capture_preview(rec, calibration=calibration)
        except Exception as exc:
            self.ud_last_preview = None
            self.ud_plot_axis.cla()
            self.ud_plot_axis.set_xlabel("Time (s)")
            self.ud_plot_axis.set_ylabel("Voltage (V)")
            self.ud_plot_axis.set_title("EMG Signal")
            self.ud_plot_info_var.set(f"Failed to process {rec.get('filename', '')}")
            self._set_ud_process_summary(f"Preview failed:\n{exc}")
            self.ud_canvas.draw_idle()
            self._log(f"Processed preview failed for {rec.get('filename', '')}: {exc}", level="ERROR")
            return

        self.ud_last_preview = preview
        self._ud_draw_capture_preview(preview)

    def _ud_redraw_current_capture(self):
        if self.ud_last_preview is not None:
            self._ud_draw_capture_preview(self.ud_last_preview)

    def _ud_plot_preview_on_axis(self, preview, axis, title_prefix=""):
        """Draw a single preview on the given matplotlib axis. No canvas.draw()."""
        view_mode = self.ud_plot_mode_var.get()
        axis.cla()

        if view_mode == "processed":
            axis.set_xlabel("Processed Time (s)")
            axis.set_ylabel("Centered Voltage (V)")
        else:
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Voltage (V)")

        start_t = None
        end_t = None
        if preview["processed_sample_count"] > 0:
            start_t = float(preview["processed_original_time_s"][0])
            end_t = float(preview["processed_original_time_s"][-1])

        for channel in preview["channels"]:
            channel_data = preview["channels_data"][channel]
            color = PLOT_COLORS.get(channel, None)

            if view_mode in ("raw", "overlay"):
                axis.plot(
                    preview["raw_time_s"],
                    channel_data["raw_voltage"],
                    label=f"{channel_display_name(channel)} raw",
                    color=color,
                    linewidth=0.9 if view_mode == "raw" else 0.8,
                    alpha=0.9 if view_mode == "raw" else 0.35,
                )

            if view_mode in ("processed", "overlay") and preview["processed_sample_count"] > 0:
                x_vals = preview["processed_time_s"] if view_mode == "processed" else preview["processed_original_time_s"]
                axis.plot(
                    x_vals,
                    channel_data["processed_voltage"],
                    label=f"{channel_display_name(channel)} processed",
                    color=color,
                    linewidth=2.0,
                )

        if view_mode in ("raw", "overlay") and start_t is not None and end_t is not None:
            axis.axvspan(start_t, end_t, color="#fde68a", alpha=0.20)

        title_suffix = {
            "raw": "Raw Signal",
            "processed": "Processed Segment",
            "overlay": "Raw + Processed Overlay",
        }.get(view_mode, "EMG Signal")
        axis.set_title(f"{title_prefix}{preview['label']}  ({preview['mode']})  |  {title_suffix}", fontsize=10)
        axis.grid(True)
        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(loc="upper right", fontsize=8)

    def _ud_draw_capture_preview(self, preview):
        # draw main on primary axis
        main_prefix = "MAIN: " if self.ud_compare_mode else ""
        self._ud_plot_preview_on_axis(preview, self.ud_plot_axis, title_prefix=main_prefix)

        # draw compare on secondary axis if active
        if self.ud_compare_mode and self.ud_compare_preview is not None and self.ud_compare_axis is not None:
            cmp_prefix = "COMPARE: "
            self._ud_plot_preview_on_axis(
                self.ud_compare_preview, self.ud_compare_axis, title_prefix=cmp_prefix,
            )

        # --- info strip + summary text (based on MAIN preview) ---
        status_text = {
            "active_segment": "active segment found",
            "full_window": "rest/full window kept",
            "skipped_no_event": "no active segment found; sample would be skipped",
        }.get(preview["status"], preview["status"])
        self.ud_plot_info_var.set(
            f"{preview['filename']}  |  {preview.get('timestamp', '')}  |  "
            f"{preview['raw_sample_count']} -> {preview['processed_sample_count']} samples  |  {status_text}"
        )

        summary_lines = [
            f"MAIN:  file={preview['filename']}  label={preview['label']}  mode={preview['mode']}",
            f"       fs={preview['fs']:.2f}Hz  event_k={preview['event_k']:.2f}  status={status_text}",
            f"       raw_samples={preview['raw_sample_count']}  processed_samples={preview['processed_sample_count']}",
        ]
        if preview.get("calibration_source"):
            summary_lines.append(f"       calibration={preview['calibration_source']}")

        for channel in preview["channels"]:
            channel_data = preview["channels_data"][channel]
            noise = channel_data["noise_std"]
            summary_lines.append(
                f"       {channel}: baseline={channel_data['baseline_mean']:.4f}V  "
                f"noise_std={(noise if noise is not None else 0.0):.4f}V  "
                f"threshold={channel_data['threshold']:.4f}V"
            )
            features = channel_data["features"]
            if features is not None:
                feature_text = ", ".join(
                    f"{name}={float(value):.4f}"
                    for name, value in zip(preview["feature_names"], features)
                )
                summary_lines.append(f"       {channel} features: {feature_text}")

        if self.ud_compare_mode and self.ud_compare_preview is not None:
            cmp_preview = self.ud_compare_preview
            cmp_status = {
                "active_segment": "active segment found",
                "full_window": "rest/full window kept",
                "skipped_no_event": "no active segment found; sample would be skipped",
            }.get(cmp_preview["status"], cmp_preview["status"])
            summary_lines.append("")
            summary_lines.append(
                f"COMPARE:  file={cmp_preview['filename']}  label={cmp_preview['label']}  "
                f"mode={cmp_preview['mode']}  user={self.ud_compare_user or '?'}"
            )
            summary_lines.append(
                f"          fs={cmp_preview['fs']:.2f}Hz  status={cmp_status}  "
                f"raw_samples={cmp_preview['raw_sample_count']}  "
                f"processed_samples={cmp_preview['processed_sample_count']}"
            )
            for channel in cmp_preview["channels"]:
                cdata = cmp_preview["channels_data"][channel]
                cnoise = cdata["noise_std"]
                summary_lines.append(
                    f"          {channel}: baseline={cdata['baseline_mean']:.4f}V  "
                    f"noise_std={(cnoise if cnoise is not None else 0.0):.4f}V  "
                    f"threshold={cdata['threshold']:.4f}V"
                )

        self._set_ud_process_summary("\n".join(summary_lines))
        try:
            self.ud_canvas.figure.tight_layout()
        except Exception:
            pass
        self.ud_canvas.draw_idle()

    # ---------------------------------------------------------------
    # User Data: Compare mode
    # ---------------------------------------------------------------
    def _ud_rebuild_axes(self):
        """Switch the figure between single-axis and 2-row split-view layouts."""
        fig = self.ud_canvas.figure
        fig.clf()
        if self.ud_compare_mode:
            self.ud_plot_axis = fig.add_subplot(211)
            self.ud_compare_axis = fig.add_subplot(212)
        else:
            self.ud_plot_axis = fig.add_subplot(111)
            self.ud_compare_axis = None
        self.ud_plot_axis.set_xlabel("Time (s)")
        self.ud_plot_axis.set_ylabel("Voltage (V)")
        self.ud_plot_axis.set_title("EMG Signal")
        if self.ud_compare_axis is not None:
            self.ud_compare_axis.set_xlabel("Time (s)")
            self.ud_compare_axis.set_ylabel("Voltage (V)")
            self.ud_compare_axis.set_title("Compare")

    def _ud_toggle_compare(self):
        if self.ud_compare_mode:
            self._ud_exit_compare()
        else:
            self._ud_enter_compare()

    def _ud_enter_compare(self):
        if self.ud_selected_record is None:
            messagebox.showinfo(
                "Select a main capture first",
                "Click a capture in the list, then click Compare.",
            )
            return
        result = self._ud_open_compare_dialog()
        if result is None:
            return
        cmp_record, cmp_user = result

        cmp_calibration = load_calibration(cmp_user) if cmp_user else None
        try:
            cmp_preview = prepare_capture_preview(cmp_record, calibration=cmp_calibration)
        except Exception as exc:
            messagebox.showerror(
                "Compare failed",
                f"Could not process {cmp_record.get('filename', '?')}:\n{exc}",
            )
            return

        self.ud_compare_record = cmp_record
        self.ud_compare_user = cmp_user
        self.ud_compare_preview = cmp_preview
        self.ud_compare_mode = True
        if self.ud_compare_btn is not None:
            self.ud_compare_btn.configure(text="Exit Compare")
        self._ud_rebuild_axes()
        # Re-render main (picks up fresh preview if needed) which will also draw compare.
        self._ud_render_selected_capture()

    def _ud_exit_compare(self):
        self.ud_compare_mode = False
        self.ud_compare_record = None
        self.ud_compare_user = None
        self.ud_compare_preview = None
        self.ud_compare_axis = None
        if self.ud_compare_btn is not None:
            self.ud_compare_btn.configure(text="Compare...")
        self._ud_rebuild_axes()
        if self.ud_selected_record is not None:
            self._ud_render_selected_capture()
        else:
            self.ud_canvas.draw_idle()

    def _ud_open_compare_dialog(self):
        """Open a modal Toplevel to pick another capture. Returns (record, user) or None."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Pick a capture to compare against")
        dialog.geometry("780x520")
        dialog.transient(self.root)
        dialog.grab_set()

        top_frame = ttk.Frame(dialog, padding=10)
        top_frame.pack(fill="both", expand=True)

        # --- left: user list ---
        user_pane = ttk.Frame(top_frame)
        user_pane.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(user_pane, text="User", font=("Helvetica", 12, "bold")).pack(anchor="w")
        cmp_user_list = tk.Listbox(user_pane, width=18, height=22, exportselection=False)
        cmp_user_list.pack(fill="y", expand=True)
        users = sorted(list_available_users("single"))
        for u in users:
            cmp_user_list.insert("end", u)

        # --- right: capture tree ---
        cap_pane = ttk.Frame(top_frame)
        cap_pane.pack(side="left", fill="both", expand=True)
        ttk.Label(cap_pane, text="Captures", font=("Helvetica", 12, "bold")).pack(anchor="w")
        cmp_cols = ("label", "mode", "channels", "timestamp")
        cmp_tree = ttk.Treeview(cap_pane, columns=cmp_cols, show="headings", height=22)
        cmp_tree.heading("label", text="Label")
        cmp_tree.heading("mode", text="Mode")
        cmp_tree.heading("channels", text="Channels")
        cmp_tree.heading("timestamp", text="Timestamp")
        cmp_tree.column("label", width=110)
        cmp_tree.column("mode", width=80)
        cmp_tree.column("channels", width=90)
        cmp_tree.column("timestamp", width=150)
        cmp_scroll = ttk.Scrollbar(cap_pane, orient="vertical", command=cmp_tree.yview)
        cmp_tree.configure(yscrollcommand=cmp_scroll.set)
        cmp_tree.pack(side="left", fill="both", expand=True)
        cmp_scroll.pack(side="right", fill="y")

        state = {"records": [], "selected_user": None}
        mode_filter = self.ud_mode_filter_var.get()

        def on_cmp_user_select(_evt):
            sel = cmp_user_list.curselection()
            if not sel:
                return
            u = cmp_user_list.get(sel[0])
            records = list_user_capture_records(u)
            if mode_filter != "all":
                records = [r for r in records if r.get("mode") == mode_filter]
            state["records"] = records
            state["selected_user"] = u
            cmp_tree.delete(*cmp_tree.get_children())
            for i, rec in enumerate(records):
                chs = ", ".join(channel_display_name(c) for c in rec.get("channels", []))
                cmp_tree.insert("", "end", iid=str(i), values=(
                    rec["label"], rec["mode"], chs, rec.get("timestamp", ""),
                ))

        cmp_user_list.bind("<<ListboxSelect>>", on_cmp_user_select)

        # preselect the currently-browsed user if any
        if self.ud_selected_user and self.ud_selected_user in users:
            pre_idx = users.index(self.ud_selected_user)
            cmp_user_list.selection_set(pre_idx)
            cmp_user_list.see(pre_idx)
            on_cmp_user_select(None)

        # --- bottom: OK / Cancel ---
        btn_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        btn_frame.pack(fill="x")

        result_holder = {"value": None}

        def on_ok():
            sel = cmp_tree.selection()
            if not sel:
                messagebox.showinfo("Pick a capture", "Select a capture row first.", parent=dialog)
                return
            idx = int(sel[0])
            if idx >= len(state["records"]):
                return
            result_holder["value"] = (state["records"][idx], state["selected_user"])
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="right", padx=(0, 6))

        cmp_tree.bind("<Double-1>", lambda _e: on_ok())
        dialog.bind("<Return>", lambda _e: on_ok())
        dialog.bind("<Escape>", lambda _e: on_cancel())

        self.root.wait_window(dialog)
        return result_holder["value"]

    def _ud_export_processed_capture(self):
        if self.ud_selected_record is None:
            messagebox.showinfo("Select capture", "Click a capture entry first.")
            return

        user_sel = self.ud_user_listbox.curselection()
        user = self.ud_user_listbox.get(user_sel[0]) if user_sel else ""
        calibration = load_calibration(user) if user else None

        try:
            exported = export_processed_capture_preview(self.ud_selected_record, calibration=calibration)
        except Exception as exc:
            self._log(f"Processed export failed: {exc}", level="ERROR")
            messagebox.showerror("Export failed", str(exc))
            return

        self._log(
            f"Processed preview exported: {exported['processed_csv']} | features: {exported['features_csv']}"
        )
        messagebox.showinfo(
            "Processed CSV Saved",
            f"Saved processed data:\n{exported['processed_csv']}\n\n"
            f"Saved feature summary:\n{exported['features_csv']}",
        )

    def _ud_delete_capture(self):
        sel = self.ud_tree.selection()
        if not sel:
            messagebox.showinfo("Select capture", "Click a capture entry first.")
            return
        idx = int(sel[0])
        if idx >= len(self.ud_records):
            return
        rec = self.ud_records[idx]
        if not messagebox.askyesno("Confirm delete",
                                    f"Delete capture '{rec['filename']}' ({rec['label']})?\n"
                                    "This will delete the CSV file(s) from disk."):
            return
        deleted = 0
        for ch, fpath in rec.get("files_by_channel", {}).items():
            try:
                fpath.unlink()
                deleted += 1
            except OSError as exc:
                self._log(f"Failed to delete {fpath}: {exc}", level="ERROR")
        self._log(f"Deleted {deleted} file(s) for capture '{rec['filename']}'")
        # refresh the list for the currently selected user
        user_sel = self.ud_user_listbox.curselection()
        if user_sel:
            self._ud_on_user_select()

    def _ud_delete_user(self):
        user_sel = self.ud_user_listbox.curselection()
        if not user_sel:
            messagebox.showinfo("Select user", "Click a user in the list first.")
            return
        user = self.ud_user_listbox.get(user_sel[0])
        if not messagebox.askyesno("Confirm delete",
                                    f"Delete ALL data for user '{user}'?\n"
                                    "This cannot be undone."):
            return
        result = clear_user_data(user)
        self._log(f"Deleted {result['deleted_files']} file(s) for user '{user}'")
        self._ud_refresh_users()

    # ---------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------
    def refresh_ports(self):
        ports = discover_serial_ports(include_pseudo=True)
        current = self.port_var.get().strip()
        if current and current not in ports:
            ports = [current] + ports
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            preferred = next(
                (
                    p for p in ports
                    if "usbserial" in p or "usbmodem" in p or p.startswith("/dev/ttys")
                ),
                ports[0],
            )
            self.port_var.set(preferred)

    def on_close(self):
        self._reset_capture_state()
        self._reset_guided_state()
        self._close_serial()
        self.root.destroy()


def main():
    root = tk.Tk()
    EMGCollectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
