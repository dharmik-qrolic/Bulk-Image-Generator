import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import json
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw

FONT_PATH_SEMIBOLD = "Inter-SemiBold.ttf"
FONT_PATH_MEDIUM = "Inter-Medium.ttf"

def warp_image_cylindrical(image, center_x, radius, curvature,
                           crop_x1, crop_y1,
                           crop_x2, crop_y2):
    roi = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    img_arr = np.array(roi)
    h, w, c = img_arr.shape
    
    yy, xx = np.mgrid[0:h, 0:w]
    
    abs_xx = xx + crop_x1
    abs_yy = yy + crop_y1
    
    src_x = xx.astype(np.float32)
    src_y = yy.astype(np.float32)
    
    dx = abs_xx - center_x
    valid_mask = np.abs(dx) < radius
    
    theta = np.zeros_like(dx, dtype=np.float32)
    theta[valid_mask] = np.arcsin(dx[valid_mask] / radius)
    
    src_x[valid_mask] = (center_x + radius * theta[valid_mask]) - crop_x1
    src_y[valid_mask] = (abs_yy[valid_mask] - curvature * (np.cos(theta[valid_mask]) - 1)) - crop_y1
    
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)
    
    warped_arr = np.zeros_like(img_arr)
    for i in range(c):
        warped_arr[..., i] = (
            img_arr[y0, x0, i] * wa +
            img_arr[y0, x1, i] * wb +
            img_arr[y1, x0, i] * wc +
            img_arr[y1, x1, i] * wd
        ).astype(np.uint8)
        
    warped_arr[~valid_mask] = 0
    
    result = Image.new("RGBA", image.size, (0, 0, 0, 0))
    result.paste(Image.fromarray(warped_arr), (crop_x1, crop_y1))
    return result

CONFIG_FILE = "editor_config.json"
CSV_FILE = "med_data.csv"

DEFAULT_FIELDS = [
    {"id": "f0", "label": "Medicine Name", "csv_column": "MEDICINES", "x": 924, "y": 908, "font_size": 46, "color": [47, 47, 47, 255], "font_weight": "SemiBold", "rotation": 0, "max_width": 0, "skew": 0, "curve": 0},
    {"id": "f1", "label": "Strength",      "csv_column": "Strength",  "x": 924, "y": 1545, "font_size": 46, "color": [47, 47, 47, 255], "font_weight": "SemiBold", "rotation": 0, "max_width": 0, "skew": 0, "curve": 0},
    {"id": "f2", "label": "Volume",        "csv_column": "Total",     "x": 924, "y": 1300, "font_size": 28, "color": [78, 78, 78, 255], "font_weight": "Medium", "rotation": 0, "max_width": 0, "skew": 0, "curve": 0},
]

DEFAULT_CONFIG = {
    "template_path": "blank_vial.png",
    "fields": list(DEFAULT_FIELDS),
    "box_width": 260,
    "box_height": 50,
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
BOX_TEXT_WIDTH = 200  # max pixel width for label text wrapping
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
        self._zoom_redraw_job = None

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
                "max_width": 0,
                "skew": 0,
                "curve": 0,
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
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        self.show_outlines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show Outlines", variable=self.show_outlines_var, command=self._draw_all).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Ready  |  Drag to pan, Mousewheel to zoom")
        ttk.Label(bar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=3)
        
        self.tab_editor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_editor, text="Label Editor")
        
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Data Editor")
        
        self._build_data_tab()

        main = ttk.PanedWindow(self.tab_editor, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=3)

        # ── canvas ──
        cf = ttk.Frame(main)
        main.add(cf, weight=1)

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
        side = ttk.Frame(main, width=380)
        main.add(side, weight=0)

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
        self.field_container_window = self.field_container_canvas.create_window((0, 0), window=self.field_frame, anchor=tk.NW, width=self.field_container_canvas.winfo_width())
        self.field_container_canvas.bind("<Configure>", lambda e: self.field_container_canvas.itemconfig(self.field_container_window, width=e.width))

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
            ("Box Width:", "box_width"),
            ("Box Height:", "box_height"),
        ]:
            row = ttk.Frame(ex)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, width=13, anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar()
            var.trace_add("write", lambda *a, k=key: self._on_extra_change(k))
            ttk.Spinbox(row, from_=-5000, to=5000, textvariable=var, width=7).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
            self.extra_vars[key] = var

        self._sync_extra_ui()

        # ── canvas bindings ──
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.canvas.bind("<Button-4>", self._on_scroll_event)
        self.canvas.bind("<Button-5>", self._on_scroll_event)
        self.canvas.bind("<MouseWheel>", self._on_scroll_event)

    # ── DATA EDITOR TAB ────────────────────────────────────────

    def _build_data_tab(self):
        self.active_csv_row = None
        
        self.tree_frame = ttk.Frame(self.tab_data)
        self.tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(self.tree_frame, yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set, show="headings")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)
        
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        
        self._load_csv_data()

    def _load_csv_data(self):
        try:
            self.df = pd.read_csv(CSV_FILE)
            self.csv_columns = list(self.df.columns)
        except Exception:
            self.df = pd.DataFrame(columns=["MEDICINES", "Strength", "Total"])
            self.csv_columns = list(self.df.columns)
            
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = self.csv_columns
        for col in self.csv_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor=tk.W)
            
        for i, row in self.df.iterrows():
            self.tree.insert("", "end", iid=str(i), values=list(row))
            
    def _on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            self.active_csv_row = None
            self._draw_all()
            return
            
        item_id = selected[0]
        vals = self.tree.item(item_id)["values"]
        self.active_csv_row = dict(zip(self.csv_columns, vals))
        self._draw_all()

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

        # Row 3: Rotation + Weight + Color
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

        # Row 4: Max Width + Skew + Remove
        row4 = ttk.Frame(inner)
        row4.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row4, text="Max W:", width=6).pack(side=tk.LEFT)
        mw_var = tk.StringVar(value=str(fld.get("max_width", 0)))
        mw_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row4, from_=0, to=2000, textvariable=mw_var, width=5).pack(side=tk.LEFT, padx=1)

        ttk.Label(row4, text="Skw:", width=4).pack(side=tk.LEFT, padx=(2, 0))
        skew_var = tk.StringVar(value=str(fld.get("skew", 0)))
        skew_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row4, from_=-200, to=200, textvariable=skew_var, width=4).pack(side=tk.LEFT, padx=1)

        ttk.Button(row4, text="X", width=2,
                   command=lambda fid=fld["id"]: self._remove_field(fid)).pack(side=tk.RIGHT, padx=1)

        # Row 5: Curve
        row5 = ttk.Frame(inner)
        row5.pack(fill=tk.X, pady=(1, 0))
        ttk.Label(row5, text="Curve:", width=6).pack(side=tk.LEFT)
        curve_var = tk.StringVar(value=str(fld.get("curve", 0)))
        curve_var.trace_add("write", lambda *a, fid=fld["id"]: self._on_field_edit(fid))
        ttk.Spinbox(row5, from_=-200, to=200, textvariable=curve_var, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(row5, text="(cylinder warp)", font=("", 8, "italic")).pack(side=tk.LEFT, padx=2)

        self.field_widgets[fld["id"]] = {
            "frame": g, "label_var": lbl_var, "col_var": col_var,
            "x_var": x_var, "y_var": y_var, "sz_var": sz_var,
            "fw_var": fw_var, "rot_var": rot_var, "color_btn": color_btn,
            "mw_var": mw_var, "skew_var": skew_var, "curve_var": curve_var,
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
        try: fld["max_width"] = int(w["mw_var"].get())
        except ValueError: pass
        try: fld["skew"] = int(w["skew_var"].get())
        except ValueError: pass
        try: fld["curve"] = int(w["curve_var"].get())
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
                "max_width": 0,
                "skew": 0,
                "curve": 0,
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
                "max_width": 0,
                "skew": 0,
                "curve": 0,
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

    def _render_image(self, fast=False):
        if self.image is None:
            return
        iw, ih = self.image.size
        sw = max(1, int(iw * self.scale))
        sh = max(1, int(ih * self.scale))
        resample_filter = Image.NEAREST if fast else Image.LANCZOS
        resized = self.image.resize((sw, sh), resample_filter)
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
        self.canvas.delete("box", "box_label", "align_line", "coord_label", "rot_label", "crop_box", "crop_label")
        if self.image is None:
            return
        self._draw_crop_box()
        self._draw_boxes()

    def _draw_crop_box(self):
        try:
            x1 = int(self.config["crop_x1"])
            y1 = int(self.config["crop_y1"])
            x2 = int(self.config["crop_x2"])
            y2 = int(self.config["crop_y2"])
            cx1, cy1 = self._img_to_canvas(x1, y1)
            cx2, cy2 = self._img_to_canvas(x2, y2)
            if getattr(self, "show_outlines_var", None) and self.show_outlines_var.get():
                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="#FF5252", width=3, dash=(6, 4),
                    tags="crop_box"
                )
                self.canvas.create_text(
                    cx1 + 8, cy1 + 8, anchor=tk.NW,
                    text="Warp / Crop Area",
                    fill="#FF5252", font=("", int(12 * max(0.5, self.scale)), "bold"),
                    tags="crop_label"
                )
        except Exception:
            pass

    def _draw_boxes(self):
        self.box_items.clear()
        
        if not hasattr(self, "font_cache"):
            self.font_cache = {}
            
        def get_font(weight, size):
            key = (weight, size)
            if key not in self.font_cache:
                path = FONT_PATH_SEMIBOLD if weight == "SemiBold" else FONT_PATH_MEDIUM
                try:
                    self.font_cache[key] = ImageFont.truetype(path, size)
                except IOError:
                    self.font_cache[key] = ImageFont.load_default()
            return self.font_cache[key]

        for i, fld in enumerate(self.config["fields"]):
            color = tuple(fld.get("color", [47, 47, 47, 255]))
            x_img = fld.get("x", self.config.get("text_align_x", 0))
            y_img = fld["y"]
            cx, cy = self._img_to_canvas(x_img, y_img)
            
            font_size = fld.get("font_size", 46)
            font_weight = fld.get("font_weight", "SemiBold")
            rotation = fld.get("rotation", 0)
            
            csv_col = fld.get("csv_column", "")
            if hasattr(self, "active_csv_row") and self.active_csv_row and csv_col in self.active_csv_row:
                text = str(self.active_csv_row[csv_col]).upper()
                if text == "NAN": text = ""
            else:
                text = fld.get("label", "Dummy Text").upper()
            
            scaled_font_size = max(1, int(font_size * self.scale))
            font = get_font(font_weight, scaled_font_size)
            max_width = fld.get("max_width", 0) * self.scale
            
            lines = [text]
            final_lines = []
            
            dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            draw = ImageDraw.Draw(dummy_img)
            
            for line in lines:
                if max_width > 0:
                    words = line.split(" ")
                    chunks = []
                    for i, word in enumerate(words):
                        if "/" in word:
                            parts = word.split("/")
                            chunks.append(parts[0])
                            for part in parts[1:]:
                                chunks.append("/" + part)
                        else:
                            chunks.append(word)
                        if i < len(words) - 1:
                            chunks.append(" ")
                            
                    current_line = ""
                    for chunk in chunks:
                        test_line = current_line + chunk
                        bbox = draw.textbbox((0, 0), test_line.strip(), font=font)
                        line_tw = bbox[2] - bbox[0]
                        if line_tw <= max_width:
                            current_line += chunk
                        else:
                            if current_line.strip():
                                final_lines.append(current_line.strip())
                            current_line = chunk
                    if current_line.strip():
                        final_lines.append(current_line.strip())
                else:
                    final_lines.append(line)
            
            line_spacing = self.config.get("line_spacing", 48) * self.scale
            measured_widths = [draw.textbbox((0, 0), l, font=font)[2] - draw.textbbox((0, 0), l, font=font)[0] for l in final_lines]
            measured_heights = [draw.textbbox((0, 0), l, font=font)[3] - draw.textbbox((0, 0), l, font=font)[1] for l in final_lines]
            
            max_measured_w = max(measured_widths) if measured_widths else 10
            w_block = max_width if max_width > 0 else max_measured_w
            w_block = max(w_block, max_measured_w)  # Prevent clipping if text overflows max_width
            
            h_block = sum(measured_heights) + max(0, len(final_lines) - 1) * line_spacing
            
            pad = 20
            txt_img = Image.new("RGBA", (int(w_block + pad * 2), int(h_block + pad * 2)), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            
            curr_y = pad
            for i, line in enumerate(final_lines):
                txt_draw.text((pad, curr_y), line, fill=color, font=font)
                curr_y += measured_heights[i] + line_spacing
                
            tw = w_block
            th = h_block
            
            skew = fld.get("skew", 0)
            if skew != 0:
                w_img, h_img = txt_img.size
                pad_y = int(abs(skew) + 10)
                dest_h = h_img + pad_y * 2
                
                dest = Image.new("RGBA", (w_img, dest_h), (0, 0, 0, 0))
                
                if skew > 0:
                    dy0, dy1 = pad_y, pad_y + h_img
                    dy2, dy3 = pad_y + h_img + skew, pad_y - skew
                else:
                    dy0, dy1 = pad_y - skew, pad_y + h_img + skew
                    dy2, dy3 = pad_y + h_img, pad_y
                
                dst_pts = [
                    (0, dy0), (0, dy1), (w_img, dy2), (w_img, dy3)
                ]
                src_pts = [
                    (0, 0), (0, h_img), (w_img, h_img), (w_img, 0)
                ]
                
                matrix = []
                for d, s in zip(dst_pts, src_pts):
                    matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
                    matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])
                A = np.array(matrix, dtype=float)
                B = np.array(src_pts, dtype=float).reshape(8)
                coeffs = np.linalg.solve(A, B)
                
                warped = txt_img.transform((w_img, dest_h), Image.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
                dest.paste(warped, (0, 0), warped)
                txt_img = dest

            curve = fld.get("curve", 0)
            if curve != 0:
                txt_img = warp_image_cylindrical(
                    txt_img,
                    center_x=txt_img.width / 2,
                    radius=self.config.get("cylinder_radius", 700),
                    curvature=curve,
                    crop_x1=0, crop_y1=0,
                    crop_x2=txt_img.width, crop_y2=txt_img.height
                )

            
            if rotation:
                txt_img = txt_img.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
                
            tk_txt = ImageTk.PhotoImage(txt_img)
            
            center_x = cx + tw / 2
            center_y = cy + th / 2
            paste_x = int(center_x - txt_img.width / 2)
            paste_y = int(center_y - txt_img.height / 2)
            
            img_id = self.canvas.create_image(paste_x, paste_y, anchor=tk.NW, image=tk_txt, tags="box")
            
            # Draw a subtle outline for interaction clarity
            rect_id = None
            if getattr(self, "show_outlines_var", None) and self.show_outlines_var.get():
                rect_id = self.canvas.create_rectangle(
                    paste_x + pad, paste_y + pad, paste_x + txt_img.width - pad, paste_y + txt_img.height - pad,
                    outline="#ffffff", width=1, dash=(2, 2), tags="box"
                )
            
            self.box_items[fld["id"]] = {"img": tk_txt, "id": img_id, "rect": rect_id}


    # ── CANVAS NAVIGATION ──────────────────────────────────────

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _pan_end(self, event):
        self.canvas.config(cursor="hand2")

    def _on_scroll_event(self, event):
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = 1 if event.delta > 0 else -1

        factor = 1.12 if delta > 0 else 1 / 1.12
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
        self._render_image(fast=True)
        self._draw_all()
        self.status_var.set(f"Zoom: {self.zoom*100:.0f}%  |  Mousewheel to zoom")

        if self._zoom_redraw_job is not None:
            self.root.after_cancel(self._zoom_redraw_job)
        self._zoom_redraw_job = self.root.after(200, self._render_high_quality)

    def _render_high_quality(self):
        self._zoom_redraw_job = None
        self._render_image(fast=False)

    def _reset_zoom(self):
        self.zoom = 1.0
        self._fit_image()
        self.status_var.set("Zoom: 100%  |  Drag to pan, Mousewheel to zoom")

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
            cmd = [sys.executable, "index.py", "--preview"]
            if hasattr(self, "active_csv_row_index") and self.active_csv_row_index is not None:
                cmd.extend(["--row", str(self.active_csv_row_index)])
            subprocess.Popen(
                cmd,
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
