import streamlit as st
from PIL import Image
import numpy as np
import io
import struct
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Tuft Designer Pro", page_icon="🧵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0d0d0d; color: #f0ece0; }
.stApp { background-color: #0d0d0d; }
h1,h2,h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
.title-block { border-left: 5px solid #c8a96e; padding: 12px 20px; margin-bottom: 30px; background: linear-gradient(90deg, #1a1a1a, #0d0d0d); }
.title-block h1 { font-size: 2.2rem; color: #c8a96e; margin: 0; }
.title-block p  { color: #888; font-family: 'DM Mono', monospace; font-size: 0.82rem; margin: 4px 0 0; }
.card { background: #161616; border: 1px solid #2a2a2a; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card h3 { color: #c8a96e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 14px; }
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; }
.stat { background: #1f1f1f; border: 1px solid #333; border-radius: 6px; padding: 10px 16px; font-family: 'DM Mono', monospace; font-size: 0.85rem; }
.stat span { color: #c8a96e; font-weight: 700; }
.result-row { background: #1a1a0d; border: 1px solid #3a3a1a; border-radius: 6px; padding: 14px; margin: 6px 0; font-family: 'DM Mono', monospace; font-size: 0.85rem; }
.result-row .label { color: #888; margin-bottom: 2px; font-size:0.78rem; }
.result-row .value { color: #e0c87a; font-weight: 700; font-size: 1rem; }
.stButton > button { background: #c8a96e !important; color: #0d0d0d !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 4px !important; padding: 10px 28px !important; font-size: 0.9rem !important; letter-spacing: 1px !important; text-transform: uppercase !important; width: 100%; }
.stDownloadButton > button { background: transparent !important; color: #c8a96e !important; border: 2px solid #c8a96e !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border-radius: 4px !important; padding: 10px 28px !important; text-transform: uppercase !important; width: 100%; }
.success-box { background: #0e1f0e; border: 1px solid #2d5a2d; border-radius: 6px; padding: 14px; color: #6fcf6f; font-family: 'DM Mono', monospace; font-size: 0.83rem; margin: 8px 0; }
.warn-box { background: #1f1800; border: 1px solid #5a4a00; border-radius: 6px; padding: 14px; color: #d4b84a; font-family: 'DM Mono', monospace; font-size: 0.83rem; }
</style>
""", unsafe_allow_html=True)

MCS_LEVELS = {
    0: {"rgb": (0,   0,   0),   "gearbox": "Idle"},
    1: {"rgb": (128, 0,   0),   "gearbox": "G1"},
    2: {"rgb": (0,   128, 0),   "gearbox": "G2"},
    3: {"rgb": (0,   0,   128), "gearbox": "G3"},
    4: {"rgb": (128, 128, 0),   "gearbox": "G4"},
    5: {"rgb": (128, 0,   128), "gearbox": "G5"},
    6: {"rgb": (0,   128, 128), "gearbox": "G6"},
    7: {"rgb": (192, 192, 192), "gearbox": "G7"},
    8: {"rgb": (255, 255, 255), "gearbox": "G8"},
}

# ─── Math Engine ───────────────────────────────────────────────────────────────

def apply_gaussian(arr, sigma):
    if sigma > 0:
        return gaussian_filter(arr.astype(np.float32), sigma=sigma)
    return arr.astype(np.float32)

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def apply_offset(arr, dx, dy):
    arr = np.roll(arr, dy, axis=0)
    arr = np.roll(arr, dx, axis=1)
    return arr

def kmeans_cluster(img_rgb, k):
    h, w, c = img_rgb.shape
    pixels = img_rgb.reshape(-1, c).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)
    return labels.reshape(h, w), centers

def compute_thread(level_arr, pile_h_mm, px_mm):
    max_lvl = level_arr.max() if level_arr.max() > 0 else 1
    f = pile_h_mm * (level_arr.astype(np.float32) / max_lvl)
    px_area = px_mm ** 2
    total_mm = float(np.sum(f) * px_area)
    per_level = {}
    for lvl in range(int(level_arr.max()) + 1):
        mask = level_arr == lvl
        cnt = int(np.sum(mask))
        length_mm = float(np.sum(f[mask]) * px_area)
        per_level[lvl] = {
            "pixels": cnt,
            "thread_m": round(length_mm / 1000, 3),
            "area_cm2": round(cnt * px_mm**2 / 100, 2),
        }
    return {"total_m": round(total_mm / 1000, 3),
            "total_cm2": round(level_arr.size * px_mm**2 / 100, 2),
            "per_level": per_level}

def build_bmp(level_arr):
    h, w = level_arr.shape
    palette = [(0,0,0)] * 256
    for i, info in MCS_LEVELS.items():
        palette[i] = info["rgb"]
    row_size = (w + 3) & ~3
    pds = row_size * h
    hs = 14 + 40 + 256 * 4
    buf = io.BytesIO()
    buf.write(b'BM')
    buf.write(struct.pack('<I', hs + pds))
    buf.write(struct.pack('<HH', 0, 0))
    buf.write(struct.pack('<I', hs))
    buf.write(struct.pack('<I', 40))
    buf.write(struct.pack('<i', w))
    buf.write(struct.pack('<i', -h))
    buf.write(struct.pack('<HH', 1, 8))
    buf.write(struct.pack('<I', 0))
    buf.write(struct.pack('<I', pds))
    buf.write(struct.pack('<ii', 2835, 2835))
    buf.write(struct.pack('<II', 256, 256))
    for r, g, b in palette:
        buf.write(struct.pack('BBBB', b, g, r, 0))
    pad = bytes(row_size - w)
    for row in level_arr:
        buf.write(row.tobytes())
        buf.write(pad)
    return buf.getvalue()

def analyze_bmp(data):
    if data[:2] != b'BM':
        return {"error": "Not a valid BMP"}
    pixel_offset = struct.unpack_from('<I', data, 10)[0]
    dib_size     = struct.unpack_from('<I', data, 14)[0]
    width        = abs(struct.unpack_from('<i', data, 18)[0])
    height_raw   = struct.unpack_from('<i', data, 22)[0]
    height       = abs(height_raw)
    bpp          = struct.unpack_from('<H', data, 28)[0]
    result = {"width": width, "height": height, "top_down": height_raw < 0,
              "bpp": bpp, "file_size": struct.unpack_from('<I', data, 2)[0]}
    if bpp == 8:
        pal_start = 14 + dib_size
        pal_entries = (pixel_offset - pal_start) // 4
        palette = []
        for i in range(min(pal_entries, 256)):
            b, g, r, _ = struct.unpack_from('BBBB', data, pal_start + i * 4)
            palette.append((r, g, b))
        result["palette"] = palette
        row_size = (width + 3) & ~3
        pixels = []
        for y in range(height):
            rs = pixel_offset + y * row_size
            pixels.extend(data[rs: rs + width])
        arr = np.array(pixels, dtype=np.uint8)
        unique, counts = np.unique(arr, return_counts=True)
        result["distribution"] = dict(zip(unique.tolist(), counts.tolist()))
        result["total_pixels"] = len(pixels)
    return result

# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <h1>🧵 Tuft Designer Pro</h1>
  <p>NedGraphics / MCS Replacement · Tufting Machine Control · Jordan</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚙️  Generate BMP", "🔍  Analyze BMP", "📊  Thread Calculator"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown('<div class="card"><h3>📁 Input Image</h3>', unsafe_allow_html=True)
        uploaded = st.file_uploader("PNG / JPG / BMP / TIFF", type=["png","jpg","jpeg","bmp","tiff","tif"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>📐 Output Size</h3>', unsafe_allow_html=True)
        cw, ch = st.columns(2)
        with cw: out_w = st.number_input("Width (px)",  8, 4096, 256, 8)
        with ch: out_h = st.number_input("Height (px)", 8, 4096, 256, 8)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>🎛️ Parameters</h3>', unsafe_allow_html=True)
        num_levels = st.slider("Levels (Gearbox stages)", 2, 9, 9)
        sigma      = st.slider("Gaussian σ (smoothing)", 0.0, 5.0, 1.0, 0.1)
        use_km     = st.checkbox("K-means color clustering")
        k_val      = st.slider("K clusters", 2, 9, 4) if use_km else 4
        dx         = st.slider("Mechanical ΔX (px)", -20, 20, 0)
        dy         = st.slider("Mechanical ΔY (px)", -20, 20, 0)
        fname      = st.text_input("Output filename", "tuft_output.bmp")
        if not fname.endswith(".bmp"): fname += ".bmp"
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h3>🖼️ Preview & Export</h3>', unsafe_allow_html=True)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Original", use_container_width=True)

            if st.button("▶  Process & Generate BMP"):
                with st.spinner("Processing..."):
                    img_r = img.resize((out_w, out_h), Image.LANCZOS)

                    if use_km:
                        rgb = np.array(img_r.convert("RGB"))
                        lmap, _ = kmeans_cluster(rgb, k_val)
                        gray = lmap.astype(np.float32) / (k_val - 1) * 255
                    else:
                        gray = np.array(img_r.convert("L"), dtype=np.float32)

                    smoothed = apply_gaussian(gray, sigma)
                    normed   = normalize(smoothed) * 255.0
                    lvl_arr  = np.floor(normed / 255.0 * (num_levels - 1) + 0.5).astype(np.uint8)
                    lvl_arr  = np.clip(lvl_arr, 0, num_levels - 1)

                    if dx != 0 or dy != 0:
                        lvl_arr = apply_offset(lvl_arr, dx, dy).astype(np.uint8)

                    bmp_bytes = build_bmp(lvl_arr)

                st.markdown('<div class="success-box">✔ BMP generated successfully</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stat-row" style="margin:12px 0;">
                  <div class="stat">Size: <span>{len(bmp_bytes):,} B</span></div>
                  <div class="stat">Dims: <span>{out_w}×{out_h}</span></div>
                  <div class="stat">Levels: <span>{num_levels}</span></div>
                  <div class="stat">σ: <span>{sigma}</span></div>
                  <div class="stat">Δ: <span>({dx},{dy})</span></div>
                </div>
                """, unsafe_allow_html=True)

                disp = (lvl_arr.astype(np.float32) / (num_levels - 1) * 255).astype(np.uint8)
                st.image(disp, caption="Level-mapped output", use_container_width=True)
                st.download_button("⬇  Download BMP", bmp_bytes, fname, "image/bmp")
        else:
            st.markdown('<div class="warn-box">← Upload an image to begin</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    ca, cb = st.columns([1, 1], gap="large")
    with ca:
        st.markdown('<div class="card"><h3>📁 Upload BMP</h3>', unsafe_allow_html=True)
        bmp_up = st.file_uploader("Upload .bmp", type=["bmp"], key="ana")
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        if bmp_up:
            info = analyze_bmp(bmp_up.read())
            if "error" in info:
                st.markdown(f'<div class="warn-box">✘ {info["error"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><h3>📋 File Info</h3>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stat-row">
                  <div class="stat">Width: <span>{info['width']} px</span></div>
                  <div class="stat">Height: <span>{info['height']} px</span></div>
                  <div class="stat">BPP: <span>{info['bpp']}</span></div>
                  <div class="stat">Size: <span>{info['file_size']:,} B</span></div>
                  <div class="stat">Top-down: <span>{'Yes' if info['top_down'] else 'No'}</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if info["bpp"] == 8 and "distribution" in info:
                    st.markdown('<div class="card"><h3>📊 Level Distribution</h3>', unsafe_allow_html=True)
                    total = info["total_pixels"]
                    for lvl, cnt in sorted(info["distribution"].items()):
                        pct = cnt / total * 100
                        gear = MCS_LEVELS.get(lvl, {}).get("gearbox", f"idx {lvl}")
                        r, g, b = MCS_LEVELS.get(lvl, {}).get("rgb", (80, 80, 80))
                        st.markdown(f"""
                        <div style="margin-bottom:10px;">
                          <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#aaa;margin-bottom:3px;">
                            <span style="display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});border-radius:2px;margin-right:6px;vertical-align:middle;"></span>
                            L{lvl} · {gear} · {cnt:,} px · {pct:.1f}%
                          </div>
                          <div style="background:#1a1a1a;border-radius:3px;height:7px;overflow:hidden;">
                            <div style="width:{min(pct,100):.1f}%;background:rgb({r},{g},{b});height:100%;"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">← Upload a BMP to analyze</div>', unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="card"><h3>🧮 Thread Consumption Calculator</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888;font-size:0.8rem;font-family:\'DM Mono\',monospace;">L = ∫∫ f(x,y) dx dy &nbsp;|&nbsp; f(x,y) = pile_height × (level / max_level)</p>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1: pile_h  = st.number_input("Pile height (mm)", 1.0, 50.0, 8.0, 0.5)
    with t2: px_size = st.number_input("Pixel size (mm)", 0.1, 10.0, 1.0, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    tc_up = st.file_uploader("Upload image or BMP", type=["bmp","png","jpg","jpeg"], key="tc")
    if tc_up:
        img_tc = Image.open(tc_up).convert("L")
        arr_tc = np.array(img_tc, dtype=np.float32)
        norm_tc = normalize(arr_tc)
        lvl_tc = np.clip(np.floor(norm_tc * 8 + 0.5).astype(np.uint8), 0, 8)
        res = compute_thread(lvl_tc, pile_h, px_size)

        st.markdown(f"""
        <div class="stat-row" style="margin:16px 0;">
          <div class="stat">Total thread: <span>{res['total_m']:,.1f} m</span></div>
          <div class="stat">Total area: <span>{res['total_cm2']:,.1f} cm²</span></div>
          <div class="stat">Pixels: <span>{lvl_tc.size:,}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Per Level Breakdown</h3>', unsafe_allow_html=True)
        for lvl, d in res["per_level"].items():
            if d["pixels"] == 0:
                continue
            gear = MCS_LEVELS.get(lvl, {}).get("gearbox", "?")
            r, g, b = MCS_LEVELS.get(lvl, {}).get("rgb", (80, 80, 80))
            st.markdown(f"""
            <div class="result-row">
              <div class="label">
                <span style="display:inline-block;width:10px;height:10px;background:rgb({r},{g},{b});border-radius:2px;margin-right:6px;vertical-align:middle;"></span>
                Level {lvl} · {gear}
              </div>
              <div class="value">{d['thread_m']:,.2f} m &nbsp; <span style="color:#555;font-size:0.78rem;">{d['area_cm2']} cm² · {d['pixels']:,} px</span></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">← Upload an image to calculate thread consumption</div>', unsafe_allow_html=True)
