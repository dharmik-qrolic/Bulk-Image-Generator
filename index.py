import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── CONFIGURATION ────────────────────────────────────────────
EDITOR_CONFIG_FILE = "editor_config.json"

DEFAULT_CONFIG = {
    "fields": [],
    "text_align_x": 924,
    "cylinder_center_x": 1112,
    "cylinder_radius": 700,
    "cylinder_curvature": 30,
    "line_spacing": 48,
    "crop_x1": 889,
    "crop_y1": 880,
    "crop_x2": 1336,
    "crop_y2": 1660,
}

config = dict(DEFAULT_CONFIG)

if os.path.exists(EDITOR_CONFIG_FILE):
    with open(EDITOR_CONFIG_FILE) as f:
        cfg = json.load(f)
    for k in DEFAULT_CONFIG:
        config[k] = cfg.get(k, DEFAULT_CONFIG[k])
    if "fields" not in cfg and "name_y" in cfg:
        fields = []
        old_keys = [
            ("Medicine Name", "name_y", "name_font_size", "name_color", "SemiBold", "MEDICINES"),
            ("Strength", "strength_y", "strength_font_size", "strength_color", "SemiBold", "Strength"),
            ("Volume", "volume_y", "volume_font_size", "volume_color", "Medium", "Total"),
        ]
        for label, key_y, key_fs, key_c, fw, col in old_keys:
            fields.append({
                "label": label,
                "csv_column": col,
                "x": cfg.get("text_align_x", 924),
                "y": cfg.get(key_y, 0),
                "font_size": cfg.get(key_fs, 36),
                "color": cfg.get(key_c, [47, 47, 47, 255]),
                "font_weight": fw,
                "rotation": 0,
            })
        config["fields"] = fields
    elif "fields" in cfg:
        config["fields"] = cfg["fields"]

# ── END CONFIGURATION ────────────────────────────────────────

import urllib.request
for font_file, url in [
    ("Inter-SemiBold.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/static/Inter-SemiBold.ttf"),
    ("Inter-Medium.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/static/Inter-Medium.ttf")
]:
    need_download = True
    if os.path.exists(font_file):
        with open(font_file, 'rb') as f:
            header = f.read(100)
            if b'<!DOCTYPE html>' not in header and b'<html' not in header:
                need_download = False
    if need_download:
        print(f"Downloading {font_file}...")
        try:
            urllib.request.urlretrieve(url, font_file)
        except Exception as e:
            print(f"Error downloading {font_file}: {e}")

csv_path = "med_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}.")

df = pd.read_csv(csv_path)

template_path = "blank_vial.png"
if os.path.exists(EDITOR_CONFIG_FILE):
    with open(EDITOR_CONFIG_FILE) as f:
        cfg = json.load(f)
    if cfg.get("template_path") and os.path.exists(cfg["template_path"]):
        template_path = cfg["template_path"]

if not os.path.exists(template_path):
    raise FileNotFoundError(f"Could not find template: {template_path}")

output_dir = "labeled_vials"
os.makedirs(output_dir, exist_ok=True)

FONT_PATH_SEMIBOLD = "Inter-SemiBold.ttf"
FONT_PATH_MEDIUM = "Inter-Medium.ttf"


def warp_image_cylindrical(image, center_x, radius, curvature,
                           crop_x1=None, crop_y1=None,
                           crop_x2=None, crop_y2=None):
    if crop_x1 is None: crop_x1 = config["crop_x1"]
    if crop_y1 is None: crop_y1 = config["crop_y1"]
    if crop_x2 is None: crop_x2 = config["crop_x2"]
    if crop_y2 is None: crop_y2 = config["crop_y2"]
    
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


def draw_labeled_vial(row, output_filename):
    img = Image.open(template_path).convert("RGBA")

    font_cache = {}
    def get_font(weight, size):
        key = (weight, size)
        if key not in font_cache:
            path = FONT_PATH_SEMIBOLD if weight == "SemiBold" else FONT_PATH_MEDIUM
            try:
                font_cache[key] = ImageFont.truetype(path, size)
            except IOError:
                font_cache[key] = ImageFont.load_default()
        return font_cache[key]

    text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    line_spacing = config.get("line_spacing", 48)

    for fld in config["fields"]:
        col = fld.get("csv_column", "")
        if col not in row.index:
            continue

        text = str(row[col]).upper()
        if text == "NAN": text = ""
        x_pos = fld.get("x", config["text_align_x"])
        y_pos = fld["y"]
        font_size = fld["font_size"]
        font_weight = fld.get("font_weight", "SemiBold")
        color = tuple(fld.get("color", [47, 47, 47, 255]))
        rotation = fld.get("rotation", 0)

        font = get_font(font_weight, font_size)
        current_y = y_pos
        max_width = fld.get("max_width", 0)

        lines = [text]
        final_lines = []
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
                    tw = bbox[2] - bbox[0]
                    if tw <= max_width:
                        current_line += chunk
                    else:
                        if current_line.strip():
                            final_lines.append(current_line.strip())
                        current_line = chunk
                if current_line.strip():
                    final_lines.append(current_line.strip())
            else:
                final_lines.append(line)

        skew = fld.get("skew", 0)

        # Measure text box size
        measured_widths = [draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in final_lines]
        measured_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in final_lines]
        
        max_measured_w = max(measured_widths) if measured_widths else 10
        w_block = max_width if max_width > 0 else max_measured_w
        w_block = max(w_block, max_measured_w)  # Prevent clipping if text overflows max_width
        
        h_block = sum(measured_heights) + max(0, len(final_lines) - 1) * line_spacing

        # Create temporary canvas for this field
        pad = 40
        temp_img = Image.new("RGBA", (int(w_block + pad * 2), int(h_block + pad * 2)), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        # Draw lines centered horizontally and top-aligned inside the padded space
        temp_y = pad
        for i, line in enumerate(final_lines):
            # We can also draw left-aligned at x=pad
            temp_draw.text((pad, temp_y), line, fill=color, font=font)
            temp_y += measured_heights[i] + line_spacing

        # Apply skew to the entire image
        if skew != 0:
            w_img, h_img = temp_img.size
            pad_y = abs(skew) + 10
            dest_h = h_img + pad_y * 2
            
            # Create a blank destination canvas
            dest = Image.new("RGBA", (w_img, dest_h), (0, 0, 0, 0))
            
            # Define mapping points for the entire image: dst_pts to src_pts
            if skew > 0:
                dy0, dy1 = pad_y, pad_y + h_img
                dy2, dy3 = pad_y + h_img + skew, pad_y - skew
            else:
                dy0, dy1 = pad_y - skew, pad_y + h_img + skew
                dy2, dy3 = pad_y + h_img, pad_y
            
            dst_pts = [
                (0, dy0),      # Top-left
                (0, dy1),      # Bottom-left
                (w_img, dy2),  # Bottom-right
                (w_img, dy3)   # Top-right
            ]
            src_pts = [
                (0, 0),            # top-left
                (0, h_img),        # bottom-left
                (w_img, h_img),    # bottom-right
                (w_img, 0)         # top-right
            ]
            
            matrix = []
            for d, s in zip(dst_pts, src_pts):
                matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
                matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])
                
            A = np.array(matrix, dtype=float)
            B = np.array(src_pts, dtype=float).reshape(8)
            coeffs = np.linalg.solve(A, B)
            
            warped = temp_img.transform((w_img, dest_h), Image.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
            dest.paste(warped, (0, 0), warped)
            temp_img = dest

        # Apply local cylinder warp/curve
        curve = fld.get("curve", 0)
        if curve != 0:
            temp_img = warp_image_cylindrical(
                temp_img,
                center_x=temp_img.width / 2,
                radius=config.get("cylinder_radius", 700),
                curvature=curve,
                crop_x1=0, crop_y1=0,
                crop_x2=temp_img.width, crop_y2=temp_img.height
            )

        # Apply rotation
        if rotation != 0:
            temp_img = temp_img.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))

        # Center alignment pasting
        cx_box = x_pos + w_block / 2
        cy_box = y_pos + h_block / 2
        paste_x = int(cx_box - temp_img.width / 2)
        paste_y = int(cy_box - temp_img.height / 2)
        text_layer.paste(temp_img, (paste_x, paste_y), temp_img)

    warped_text = warp_image_cylindrical(
        text_layer,
        center_x=config["cylinder_center_x"],
        radius=config["cylinder_radius"],
        curvature=config["cylinder_curvature"],
        crop_x1=config["crop_x1"], crop_y1=config["crop_y1"],
        crop_x2=config["crop_x2"], crop_y2=config["crop_y2"],
    )

    warped_text = warped_text.filter(ImageFilter.GaussianBlur(radius=0.5))
    final_img = Image.alpha_composite(img, warped_text)
    final_img.save(os.path.join(output_dir, output_filename), "PNG")


PREVIEW_MODE = "--preview" in sys.argv

rows = list(df.iterrows())

if PREVIEW_MODE:
    if "--row" in sys.argv:
        try:
            idx = int(sys.argv[sys.argv.index("--row") + 1])
            rows = [rows[idx]]
        except (ValueError, IndexError):
            rows = rows[:1]
    else:
        rows = rows[:1]
    print("Preview mode: generating single preview entry")

print("Starting label generation...")
for _, row in rows:
    parts = []
    for fld in config["fields"]:
        col = fld.get("csv_column", "")
        if col in row.index:
            parts.append(str(row[col]))
    safe_parts = []
    for p in parts:
        p = p.replace("/", "_").replace(" ", "_")
        safe_parts.append(p)
    filename = "_".join(safe_parts) + ".png" if safe_parts else "output.png"

    print(f"Generating: {filename}")
    draw_labeled_vial(row, filename)

print(f"\nDone. Images saved to '{output_dir}'.")
