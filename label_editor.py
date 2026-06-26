import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import json
import os
import sys
import pandas as pd
from PIL import Image, ImageTk

CONFIG_FILE = "editor_config.json"
CSV_FILE = "med_data.csv"

DEFAULT_FIELDS = [
    {"id": "f0", "label": "Medicine Name", "csv_column": "MEDICINES", "x": 924, "y": 908, "font_size": 46, "color": [47, 47, 47, 255], "font_weight": "SemiBold", "rotation": 0},
    {"id": "f1", "label": "Strength",      "csv_column": "Strength",  "x": 924, "y": 1545, "font_size": 46, "color": [47, 47, 47, 255], "font_weight": "SemiBold", "rotation": 0},
    {"id": "f2", "label": "Volume",        "csv_column": "Total",     "x": 924, "y": 1300, "font_size": 28, "color": [78, 78, 78, 255], "font_weight": "Medium", "rotation": 0},
]

DEFAULT_CONFIG = {
    "template_path": "blank_vial.png",
    "fields": list(DEFAULT_FIELDS),
    "text_align_x": 924,
    "line_spacing": 48,
    "crop_x1": 889,
    "crop_y1": 880,
    "crop_x2": 1336,
    "crop_y2": 1660,
    "cylinder_center_x": 1112,
    "cylinder_radius": 700,
    "cylinder_curvature": 30,
}

CANVAS_W = 1000
CANVAS_H = 750
BOX_W = 260
BOX_H = 50
FIELD_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4", "#795548", "#607D8B"]

FONT_WEIGHTS = ["SemiBold", "Medium"]


class LabelEditor:
    def __init__(self):
        self.config = self._load_config()
        self.image = None
        self.tk_image = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.box_items = {}
        self.field_widgets = {}
        self.zoom = 1.0
        self.base_scale = 1.0
        self.csv_columns = self._get_csv_columns()
        self._next_id = self._find_next_id()

        self.root = tk.Tk()
        self.root.title("Vial Label Editor")
        self.root.geometry("1400x800+100+50")

        self._build_ui()
        self._load_image(self.config["template_path"])

    def _get_csv_columns(self):
        try:
            df = pd.read_csv(CSV_FILE)
            return list(df.columns)
        except Exception:
            return ["MEDICINES", "Strength", "Total"]

    def _find_next_id(self):
        n = 0
        for f in self.config["fields"]:
            fid = f.get("id", "")
            if fid.startswith("f") and fid[1:].isdigit():
                n = max(n, int(fid[1:]) + 1)
        return n

    def _next_field_id(self):
        fid = f"f{self._next_id}"
        self._next_id += 1
        return fid

    # ── CONFIG ─────────────────────────────────────────────────

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            if "fields" not in cfg:
                cfg = self._migrate_old_config(cfg)
            for k in DEFAULT_CONFIG:
                cfg.setdefault(k, DEFAULT_CONFIG[k])
            return cfg
        return dict(DEFAULT_CONFIG)

    def _migrate_old_config(self, old):
        old_keys = {
            "name":     {"key_y": "name_y",     "key_font": "name_font_size", "key_color": "name_color"},
            "strength": {"key_y": "strength_y",  "key_font": "strength_font_size", "key_color": "strength_color"},
            "volume":   {"key_y": "volume_y",    "key_font": "volume_font_size", "key_color": "volume_color"},
        }
        new = {"fields": []}
        csv_cols = self._get_csv_columns()
        for i, (fid, keys) in enumerate(old_keys.items()):
            col_map = {"name": "MEDICINES", "strength": "Strength", "volume": "Total"}
            field = {
                "id": f"f{i}",
                "label": fid.capitalize(),
                "csv_column": col_map.get(fid, csv_cols[i] if i < len(csv_cols) else ""),
                "x": old.get("text_align_x", 924),
                "y": old.get(keys["key_y"], DEFAULT_FIELDS[i]["y"]),
                "font_size": old.get(keys["key_font"], DEFAULT_FIELDS[i]["font_size"]),
                "color": old.get(keys["key_color"], list(DEFAULT_FIELDS[i]["color"])),
                "font_weight": "SemiBold" if fid != "volume" else "Medium",
                "rotation": 0,
            }
            new["fields"].append(field)
        for k in DEFAULT_CONFIG:
            if k != "fields":
                new[k] = old.get(k, DEFAULT_CONFIG[k])
        return new

    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    # ── UI ─────────────────────────────────────────────────────

    def _build_ui(self):
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=3)

        ttk.Button(bar, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=1)
        ttk.Button(bar, text="Save Config", command=self._save).pack(side=tk.LEFT, padx=1)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(bar, text="Preview (1)", command=self._preview).pack(side=tk.LEFT, padx=1)
        ttk.Button(bar, text="Generate All", command=self._generate_all).pack(side=tk.LEFT, padx=1)
        ttk.Button(bar, text="Open Output", command=self._open_output).pack(side=tk.LEFT, padx=1)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(bar, text="Reset Zoom", command=self._reset_zoom).pack(side=tk.LEFT, padx=1)

        self.status_var = tk.StringVar(value="Ready  |  Drag to pan, Mousewheel to zoom")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6)

        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=3)

        # ── canvas ──
        cf = ttk.Frame(main)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        hbar = ttk.Scrollbar(cf, orient=tk.HORIZONTAL)
        vbar = ttk.Scrollbar(cf, orient=tk.VERTICAL)

        self.canvas = tk.Canvas(
            cf, width=CANVAS_W, height=CANVAS_H,
            bg="#1e1e1e", cursor="hand2",
            xscrollcommand=hbar.set, yscrollcommand=vbar.set,
        )
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── side panel ──
        side = ttk.Frame(main, width=300)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)

        # Top section: fields list (scrollable)
        top_frame = ttk.Frame(side)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(top_frame, text="Fields", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 2))

        self.field_container_canvas = tk.Canvas(top_frame, width=290, highlightthickness=0)
        self.field_scrollbar = ttk.Scrollbar(top_frame, orient=tk.VERTICAL, command=self.field_container_canvas.yview)
        self.field_container_canvas.configure(yscrollcommand=self.field_scrollbar.set)

        self.field_container_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.field_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.field_frame = ttk.Frame(self.field_container_canvas)
        self.field_frame.bind("<Configure>", lambda e: self.field_container_canvas.configure(scrollregion=self.field_container_canvas.bbox("all")))
        self.field_container_window = self.field_container_canvas.create_window((0, 0), window=self.field_frame, anchor=tk.NW, width=280)

        self._rebuild_field_editors()

        # Bottom section: add field + warp/crop (fixed)
        bottom_frame = ttk.Frame(side)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        add_frame = ttk.Frame(bottom_frame)
        add_frame.pack(fill=tk.X, pady=(2, 2))
        self.add_var = tk.StringVar(value="+ Add Field")
        add_choices = [f"From column: {col}" for col in self.csv_columns] + ["Custom field..."]
        self.add_combo = ttk.Combobox(add_frame, textvariable=self.add_var,
            values=add_choices, state="readonly")
        self.add_combo.pack(fill=tk.X)
        self.add_combo.bind("<<ComboboxSelected>>", self._on_add_selected)

        ex = ttk.LabelFrame(bottom_frame, text="Warp & Crop", padding=3)
        ex.pack(fill=tk.X, pady=(0, 4))

        self.extra_vars = {}
        for label, key in [
            ("Line Spacing:", "line_spacing"),
            ("Crop X1:", "crop_x1"), ("Crop Y1:", "crop_y1"),
            ("Crop X2:", "crop_x2"), ("Crop Y2:", "crop_y2"),
            ("Cyl Center X:", "cylinder_center_x"),
            ("Cyl Radius:", "cylinder_radius"),
            ("Cyl Curvature:", "cylinder_curvature"),
        ]:
            row = ttk.Frame(ex)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, width=13, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar()
            var.trace_add("write", lambda *a, k=key: self._on_extra_change(k))
            ttk.Entry(row, textvariable=var, width=7).pack(side=tk.LEFT, padx=1)
            self.extra_vars[key] = var

        self._sync_extra_ui()

        # ── canvas bindings ──
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.canvas.bind("<Button-4>", self._on_scroll_event)
        self.canvas.bind("<Button-5>", self._on_scroll_event)
        self.canvas.bind("<MouseWheel>", self._on_scroll_event)

    # ── FIELD EDITORS ──────────────────────────────────────────

    def _rebuild_field_editors(self):
        for w in self.field_frame.winfo_children():
            w.destroy()
        self.field_widgets.clear()
        for i, fld in enumerate(self.config["fields"]):
            self._build_field_editor(i, fld)

    def _build_field_editor(self, i, fld):
        color = FIELD_COLORS[i % len(FIELD_COLORS)]
        g = ttk.LabelFrame(self.field_frame, text=fld["label"], padding=3)
        g.pack(fill=tk.X, pady=1)

        inner = ttk.Frame(g)
        inner.pack(fill=tk.X)

        # Row 1: Label + CSV Column
        row1 = ttk.Frame(inner)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="L:", width=2).pack(side=tk.LEFT)
        lbl_var = tk.StringVar(value=fld["label"])
        lbl_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Entry(row1, textvariable=lbl_var, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Label(row1, text="Col:", width=3).pack(side=tk.LEFT, padx=(4, 0))
        col_var = tk.StringVar(value=fld.get("csv_column", ""))
        col_combo = ttk.Combobox(row1, textvariable=col_var, values=self.csv_columns, width=10, state="readonly")
        col_combo.pack(side=tk.LEFT, padx=1)
        col_combo.bind("<<ComboboxSelected>>", lambda e, fid=fld["id"]: self._on_field_edit(fid))

        # Row 2: X + Y + Size
        row2 = ttk.Frame(inner)
        row2.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row2, text="X:", width=2).pack(side=tk.LEFT)
        x_var = tk.StringVar(value=str(fld.get("x", self.config["text_align_x"])))
        x_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row2, from_=0, to=5000, textvariable=x_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Label(row2, text="Y:", width=2).pack(side=tk.LEFT, padx=(4, 0))
        y_var = tk.StringVar(value=str(fld["y"]))
        y_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row2, from_=0, to=5000, textvariable=y_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Label(row2, text="Sz:", width=2).pack(side=tk.LEFT, padx=(4, 0))
        sz_var = tk.StringVar(value=str(fld["font_size"]))
        sz_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row2, from_=6, to=200, textvariable=sz_var, width=5).pack(side=tk.LEFT, padx=1)

        # Row 3: Rotation + Weight + Color + Remove
        row3 = ttk.Frame(inner)
        row3.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row3, text="Rot:", width=3).pack(side=tk.LEFT)
        rot_var = tk.StringVar(value=str(fld.get("rotation", 0)))
        rot_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row3, from_=0, to=360, textvariable=rot_var, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(row3, text="W:", width=2).pack(side=tk.LEFT, padx=(4, 0))
        fw_var = tk.StringVar(value=fld.get("font_weight", "SemiBold"))
        fw_combo = ttk.Combobox(row3, textvariable=fw_var, values=FONT_WEIGHTS, width=7, state="readonly")
        fw_combo.pack(side=tk.LEFT, padx=1)
        fw_combo.bind("<<ComboboxSelected>>", lambda e, fid=fld["id"]: self._on_field_edit(fid))

        color_hex = "#%02x%02x%02x" % tuple(fld["color"][:3])
        color_btn = tk.Button(row3, bg=color_hex, width=2, relief=tk.FLAT,
                              command=lambda fid=fld["id"]: self._pick_color(fid))
        color_btn.pack(side=tk.LEFT, padx=(4, 1))

        ttk.Button(row3, text="X", width=2,
                   command=lambda fid=fld["id"]: self._remove_field(fid)).pack(side=tk.RIGHT, padx=1)

        self.field_widgets[fld["id"]] = {
            "frame": g, "label_var": lbl_var, "col_var": col_var,
            "x_var": x_var, "y_var": y_var, "sz_var": sz_var,
            "fw_var": fw_var, "rot_var": rot_var, "color_btn": color_btn,
        }

    def _on_field_edit(self, fid):
        w = self.field_widgets.get(fid)
        if not w:
            return
        fld = next((f for f in self.config["fields"] if f["id"] == fid), None)
        if not fld:
            return
        fld["label"] = w["label_var"].get()
        fld["csv_column"] = w["col_var"].get()
        try: fld["x"] = int(w["x_var"].get())
        except ValueError: pass
        try: fld["y"] = int(w["y_var"].get())
        except ValueError: pass
        try: fld["font_size"] = int(w["sz_var"].get())
        except ValueError: pass
        try: fld["rotation"] = int(w["rot_var"].get())
        except ValueError: pass
        fld["font_weight"] = w["fw_var"].get()
        w["frame"].configure(text=fld["label"])
        # Warn if column doesn't exist in CSV
        col = fld["csv_column"]
        if col and col not in self.csv_columns and not hasattr(self, '_col_warned'):
            self._col_warned = True
            messagebox.showwarning("Unknown Column",
                f"Column '{col}' not found in CSV.\nAvailable: {', '.join(self.csv_columns)}\nThis field will be skipped during generation.")
        self._draw_all()

    def _pick_color(self, fid):
        fld = next((f for f in self.config["fields"] if f["id"] == fid), None)
        if not fld:
            return
        current = tuple(fld["color"][:3])
        hex_current = "#%02x%02x%02x" % current
        result = colorchooser.askcolor(hex_current, title=f"Color for {fld['label']}")
        if result and result[0]:
            r, g, b = [int(x) for x in result[0]]
            fld["color"] = [r, g, b, 255]
            w = self.field_widgets.get(fid)
            if w:
                w["color_btn"].configure(bg=result[1])
            self._draw_all()

    def _on_add_selected(self, event):
        val = self.add_var.get()
        if val.startswith("From column: "):
            col = val[len("From column: "):]
            self._add_field_from_column(col)
        elif val == "Custom field...":
            self._add_field_from_column("")
        self.add_var.set("+ Add Field")

    def _add_field_from_column(self, column):
        if column:
            if column not in self.csv_columns:
                messagebox.showwarning("Unknown Column",
                    f"Column '{column}' not found in CSV.\nAvailable: {', '.join(self.csv_columns)}")
                return
            label = column.replace("_", " ").title()
            fld = {
                "id": self._next_field_id(),
                "label": label,
                "csv_column": column,
                "x": self.config["text_align_x"],
                "y": 1000,
                "font_size": 36,
                "color": [47, 47, 47, 255],
                "font_weight": "SemiBold",
                "rotation": 0,
            }
        else:
            fld = {
                "id": self._next_field_id(),
                "label": f"Field {self._next_id - 1}",
                "csv_column": "",
                "x": self.config["text_align_x"],
                "y": 1000,
                "font_size": 36,
                "color": [47, 47, 47, 255],
                "font_weight": "SemiBold",
                "rotation": 0,
            }
        # Ensure unique ID
        existing_ids = {f["id"] for f in self.config["fields"]}
        while fld["id"] in existing_ids:
            fld["id"] = self._next_field_id()
        self.config["fields"].append(fld)
        self._rebuild_field_editors()
        self._draw_all()
        # Scroll to bottom so the new field's controls are visible
        self.field_container_canvas.update_idletasks()
        self.field_container_canvas.yview_moveto(1.0)

    def _remove_field(self, fid):
        if len(self.config["fields"]) <= 1:
            messagebox.showwarning("Cannot Remove", "Must have at least one field.")
            return
        self.config["fields"] = [f for f in self.config["fields"] if f["id"] != fid]
        self._rebuild_field_editors()
        self._draw_all()

    # ── IMAGE ──────────────────────────────────────────────────

    def _load_image(self, path):
        if not os.path.exists(path):
            self.status_var.set(f"File not found: {path}")
            return
        self.config["template_path"] = path
        self.image = Image.open(path).convert("RGBA")
        self._fit_image()

    def _fit_image(self):
        self.canvas.delete("all")
        self.box_items.clear()
        iw, ih = self.image.size
        self.base_scale = min(CANVAS_W / iw, CANVAS_H / ih)
        self.scale = self.base_scale * self.zoom
        self.offset_x = (CANVAS_W - iw * self.scale) / 2
        self.offset_y = (CANVAS_H - ih * self.scale) / 2
        self._render_image()
        self._draw_all()

    def _render_image(self):
        if self.image is None:
            return
        iw, ih = self.image.size
        sw = max(1, int(iw * self.scale))
        sh = max(1, int(ih * self.scale))
        resized = self.image.resize((sw, sh), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("img")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image, tags="img")
        self.canvas.lower("img")

    def _img_to_canvas(self, x, y):
        return (self.offset_x + x * self.scale, self.offset_y + y * self.scale)

    def _canvas_to_img(self, cx, cy):
        return ((cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale)

    # ── DRAWING ────────────────────────────────────────────────

    def _draw_all(self):
        self.canvas.delete("box", "box_label", "align_line", "coord_label", "rot_label")
        if self.image is None:
            return
        self._draw_boxes()
        self._draw_coord_labels()

    def _draw_boxes(self):
        self.box_items.clear()
        for i, fld in enumerate(self.config["fields"]):
            color = FIELD_COLORS[i % len(FIELD_COLORS)]
            x_img = fld.get("x", self.config["text_align_x"])
            y_img = fld["y"]
            cx, cy = self._img_to_canvas(x_img, y_img)
            bw = BOX_W * self.scale
            bh = BOX_H * self.scale

            rect = self.canvas.create_rectangle(
                cx, cy, cx + bw, cy + bh,
                fill=color, outline=color, width=2,
                tags="box",
            )
            lbl = self.canvas.create_text(
                cx + 4, cy + bh / 2, anchor=tk.W,
                text=fld["label"], fill="white",
                font=("", int(9 * self.scale), "bold"),
                tags="box_label",
            )
            # Rotation indicator
            rot = fld.get("rotation", 0)
            if rot:
                self.canvas.create_text(
                    cx + bw - 4, cy + 4, anchor=tk.NE,
                    text=f"{rot}\u00b0", fill="#FFEB3B",
                    font=("", int(8 * self.scale)),
                    tags=("rot_label",),
                )
            # X guide line
            self.canvas.create_line(
                cx, cy - 20 * self.scale, cx, cy + bh + 10 * self.scale,
                fill=color, width=1, dash=(3, 2),
                tags="coord_label",
            )
            # X label
            self.canvas.create_text(
                cx, cy - 22 * self.scale, anchor=tk.S,
                text=f"X:{x_img}", fill=color,
                font=("", int(8 * self.scale)),
                tags="coord_label",
            )
            self.box_items[fld["id"]] = {"rect": rect, "text": lbl}

    def _draw_coord_labels(self):
        for i, fld in enumerate(self.config["fields"]):
            color = FIELD_COLORS[i % len(FIELD_COLORS)]
            x_img = fld.get("x", self.config["text_align_x"]) + BOX_W + 8
            y_img = fld["y"]
            cx, cy = self._img_to_canvas(x_img, y_img)
            rot = fld.get("rotation", 0)
            label = f"Y:{y_img}"
            if rot:
                label += f" R:{rot}"
            self.canvas.create_text(
                cx, cy + BOX_H * self.scale / 2, anchor=tk.W,
                text=label,
                fill=color,
                font=("", int(8 * self.scale)),
                tags="coord_label",
            )

    # ── CANVAS NAVIGATION ──────────────────────────────────────

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _pan_end(self, event):
        self.canvas.config(cursor="hand2")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _reset_zoom(self):
        self.zoom = 1.0
        self._fit_image()
        self.status_var.set("Zoom: 100%  |  Drag to pan, Mousewheel to zoom")

    def _on_zoom(self, event):
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        new_zoom = max(0.1, min(10.0, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        # Keep the point under cursor stationary
        mx, my = event.x, event.y
        img_x = (mx - self.offset_x) / self.scale
        img_y = (my - self.offset_y) / self.scale
        self.zoom = new_zoom
        self.scale = self.base_scale * self.zoom
        self.offset_x = mx - img_x * self.scale
        self.offset_y = my - img_y * self.scale
        self._render_image()
        self._draw_all()
        self.status_var.set(f"Zoom: {self.zoom*100:.0f}%  |  Ctrl+Wheel to zoom")

    # ── EXTRA UI ───────────────────────────────────────────────

    def _on_extra_change(self, key):
        var = self.extra_vars[key]
        try:
            self.config[key] = int(var.get())
        except ValueError:
            pass
        self._draw_all()

    def _sync_extra_ui(self):
        for key, var in self.extra_vars.items():
            var.set(str(self.config[key]))

    # ── ACTIONS ────────────────────────────────────────────────

    def _check_field_issues(self):
        warnings = []
        for fld in self.config["fields"]:
            col = fld.get("csv_column", "")
            if col and col not in self.csv_columns:
                warnings.append(f"  - '{fld['label']}': column '{col}' not in CSV")
        if warnings:
            return "Some fields have issues:\n" + "\n".join(warnings)
        return ""

    def _save(self):
        self._save_config()
        self.status_var.set("Config saved!")

    def _preview(self):
        issues = self._check_field_issues()
        if issues:
            if not messagebox.askyesno("Field Issues", f"{issues}\n\nContinue anyway?"):
                return
        self._save_config()
        try:
            import subprocess
            subprocess.Popen(
                [sys.executable, "index.py", "--preview"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            self.status_var.set("Preview launched...")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate_all(self):
        issues = self._check_field_issues()
        if issues:
            if not messagebox.askyesno("Field Issues", f"{issues}\n\nContinue anyway?"):
                return
        self._save_config()
        self.status_var.set("Generating all labels...")
        self.root.update_idletasks()

        pw = tk.Toplevel(self.root)
        pw.title("Generating Labels")
        pw.geometry("380x110+550+400")
        pw.resizable(False, False)
        pw.transient(self.root)
        pw.grab_set()

        ttk.Label(pw, text="Generating labels, please wait...", font=("", 11)).pack(pady=(12, 2))
        prog = ttk.Progressbar(pw, mode="indeterminate", length=300)
        prog.pack(pady=6)
        prog.start(10)

        try:
            import subprocess, threading
            result = []

            def run():
                try:
                    p = subprocess.Popen(
                        [sys.executable, "index.py"],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True,
                    )
                    out, err = p.communicate()
                    result.append((p.returncode, out, err))
                except Exception as e:
                    result.append((1, "", str(e)))
                pw.after(0, pw.destroy)

            threading.Thread(target=run, daemon=True).start()
            self.root.wait_window(pw)

            retcode, stdout, stderr = result[0] if result else (1, "", "Failed to start process")
            if retcode == 0:
                lines = [l for l in stdout.split("\n") if l.strip()]
                msg = f"Done! {len(lines)-1} labels generated.\n\nOutput:\n" + "\n".join(lines[-6:])
                answer = messagebox.askyesno(
                    "Complete",
                    f"{msg}\n\nOpen output folder?",
                )
                if answer:
                    import subprocess as sp
                    sp.Popen(["xdg-open", os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_vials")])
                self.status_var.set("Generation complete!")
            else:
                messagebox.showerror("Generation Error", stderr or stdout)
                self.status_var.set("Generation failed!")
        except Exception as e:
            if pw.winfo_exists():
                pw.destroy()
            messagebox.showerror("Error", str(e))
            self.status_var.set("Generation failed!")

    def _open_output(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_vials")
        if os.path.isdir(path):
            import subprocess
            subprocess.Popen(["xdg-open", path])
        else:
            messagebox.showinfo("No Output", "No output folder yet. Generate labels first.")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select blank vial image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if path:
            self._load_image(path)
            self._sync_extra_ui()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    LabelEditor().run()
