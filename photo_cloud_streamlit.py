#!/usr/bin/env python3
"""
Photo Cloud Generator - Application Streamlit
Support étendu des formats d'image: JPEG, PNG, GIF, WebP, BMP, TIFF, HEIC, RAW, etc.

Installation des dépendances:
    pip install streamlit pillow pillow-heif rawpy imageio numpy

Lancer avec: streamlit run photo_cloud_streamlit.py
"""

import streamlit as st
import math
import random
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# =============================================================================
# CORRECTION D'ORIENTATION EXIF
# =============================================================================

def fix_orientation(img: Image.Image) -> Image.Image:
    """Corrige l'orientation de l'image selon les métadonnées EXIF."""
    try:
        exif = img.getexif()
        if not exif:
            return img
        orientation = exif.get(274)  # 274 = Orientation tag
        if orientation is None:
            return img
        if orientation == 2:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif orientation == 4:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = img.transpose(Image.Transpose.ROTATE_90)
        elif orientation == 6:
            img = img.transpose(Image.Transpose.ROTATE_270)
        elif orientation == 7:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            img = img.transpose(Image.Transpose.ROTATE_270)
        elif orientation == 8:
            img = img.transpose(Image.Transpose.ROTATE_90)
        return img
    except:
        return img

# Tentative d'import des bibliothèques optionnelles pour formats étendus
HEIF_SUPPORT = False
RAW_SUPPORT = False
IMAGEIO_SUPPORT = False

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    pass

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    pass

try:
    import imageio.v3 as iio
    IMAGEIO_SUPPORT = True
except ImportError:
    try:
        import imageio as iio
        IMAGEIO_SUPPORT = True
    except ImportError:
        pass

try:
    import numpy as np
    NUMPY_SUPPORT = True
except ImportError:
    NUMPY_SUPPORT = False


# =============================================================================
# CONFIGURATION DES FORMATS
# =============================================================================

PILLOW_EXTENSIONS = {
    '.jpg', '.jpeg', '.jpe', '.jfif',
    '.png', '.gif', '.bmp', '.dib',
    '.tiff', '.tif', '.webp', '.ico',
    '.ppm', '.pgm', '.pbm', '.pnm',
    '.pcx', '.tga', '.icb', '.vda', '.vst',
    '.dds', '.sgi', '.rgb', '.rgba', '.bw',
    '.j2k', '.j2p', '.jpx', '.jp2',
    '.eps', '.ps', '.im', '.msp', '.xbm',
    '.palm', '.pdf', '.psd', '.qoi',
}

HEIF_EXTENSIONS = {'.heif', '.heifs', '.heic', '.heics', '.avci', '.avcs', '.avif', '.avifs'}

RAW_EXTENSIONS = {
    '.cr2', '.cr3', '.crw', '.nef', '.nrw', '.arw', '.srf', '.sr2',
    '.raf', '.orf', '.rw2', '.raw', '.pef', '.ptx', '.srw', '.x3f',
    '.rwl', '.dng', '.dcr', '.k25', '.kdc', '.mrw', '.erf', '.iiq',
    '.mef', '.3fr', '.fff', '.riff',
}

IMAGEIO_EXTENSIONS = {'.exr', '.hdr', '.rgbe', '.pfm', '.fits', '.fts'}


def get_all_supported_extensions():
    """Retourne toutes les extensions supportées."""
    extensions = set(PILLOW_EXTENSIONS)
    if HEIF_SUPPORT:
        extensions.update(HEIF_EXTENSIONS)
    if RAW_SUPPORT:
        extensions.update(RAW_EXTENSIONS)
    if IMAGEIO_SUPPORT:
        extensions.update(IMAGEIO_EXTENSIONS)
    return extensions


def get_supported_formats_info():
    """Retourne les informations sur les formats supportés."""
    info = ["**Formats supportés:**"]
    info.append("- ✅ JPEG, PNG, GIF, WebP, BMP, TIFF, TGA, PCX, PPM...")
    
    if HEIF_SUPPORT:
        info.append("- ✅ HEIF/HEIC, AVIF (pillow-heif)")
    else:
        info.append("- ⚠️ HEIF/HEIC: `pip install pillow-heif`")
    
    if RAW_SUPPORT:
        info.append("- ✅ RAW: CR2, CR3, NEF, ARW, RAF, DNG... (rawpy)")
    else:
        info.append("- ⚠️ RAW: `pip install rawpy`")
    
    if IMAGEIO_SUPPORT:
        info.append("- ✅ HDR: EXR, HDR, PFM (imageio)")
    else:
        info.append("- ⚠️ HDR: `pip install imageio`")
    
    return "\n".join(info)


# =============================================================================
# CHARGEMENT DES IMAGES
# =============================================================================

def load_image_pillow(file_data: bytes) -> Image.Image:
    """Charge une image avec Pillow et corrige l'orientation EXIF."""
    img = Image.open(io.BytesIO(file_data))
    img = fix_orientation(img)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGBA')
    return img


def load_image_raw(file_data: bytes, ext: str) -> Image.Image:
    """Charge une image RAW avec rawpy."""
    if not RAW_SUPPORT:
        raise ImportError("rawpy non installé")
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name
    
    try:
        with rawpy.imread(tmp_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=8)
        img = Image.fromarray(rgb)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img
    finally:
        os.unlink(tmp_path)


def load_image_imageio(file_data: bytes) -> Image.Image:
    """Charge une image avec imageio (HDR, EXR, etc.)."""
    if not IMAGEIO_SUPPORT or not NUMPY_SUPPORT:
        raise ImportError("imageio ou numpy non installé")
    
    img_array = iio.imread(io.BytesIO(file_data))
    
    if img_array.dtype in (np.float32, np.float64, np.float16):
        img_array = img_array / (1 + img_array)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    elif img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGBA')
    return img


def load_image_auto(uploaded_file) -> Image.Image:
    """Charge une image en détectant automatiquement le format."""
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()
    file_data = uploaded_file.getvalue()
    
    errors = []
    
    if ext in HEIF_EXTENSIONS and HEIF_SUPPORT:
        try:
            return load_image_pillow(file_data)
        except Exception as e:
            errors.append(f"HEIF: {e}")
    
    if ext in RAW_EXTENSIONS:
        try:
            return load_image_raw(file_data, ext)
        except Exception as e:
            errors.append(f"RAW: {e}")
    
    if ext in IMAGEIO_EXTENSIONS:
        try:
            return load_image_imageio(file_data)
        except Exception as e:
            errors.append(f"ImageIO: {e}")
    
    try:
        return load_image_pillow(file_data)
    except Exception as e:
        errors.append(f"Pillow: {e}")
    
    if IMAGEIO_SUPPORT:
        try:
            return load_image_imageio(file_data)
        except Exception as e:
            errors.append(f"ImageIO fallback: {e}")
    
    raise ValueError(f"Impossible de charger {filename}:\n" + "\n".join(errors))


# =============================================================================
# FONCTIONS DE TRAITEMENT D'IMAGE
# =============================================================================

def resize_to_fit(img: Image.Image, max_size: tuple) -> Image.Image:
    img = img.copy()
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def resize_cover(img: Image.Image, target_size: tuple) -> Image.Image:
    target_w, target_h = target_size
    scale = max(target_w / img.width, target_h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def add_rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
    result = img.copy()
    result.putalpha(mask)
    return result


def add_border(img: Image.Image, width: int, color: tuple) -> Image.Image:
    if width <= 0:
        return img
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    bordered = Image.new('RGBA', (img.width + width*2, img.height + width*2), color)
    bordered.paste(img, (width, width), img)
    return bordered


def add_shadow(img: Image.Image, offset: tuple = (8, 8), blur: int = 15) -> Image.Image:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    padding = blur * 2 + max(abs(offset[0]), abs(offset[1]))
    shadow = Image.new('RGBA', (img.width + padding*2, img.height + padding*2), (0,0,0,0))
    shadow_shape = Image.new('RGBA', img.size, (0, 0, 0, 100))
    shadow_shape.putalpha(img.split()[3])
    shadow.paste(shadow_shape, (padding + offset[0], padding + offset[1]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    shadow.paste(img, (padding, padding), img)
    return shadow


# =============================================================================
# GÉNÉRATEURS DE POSITIONS
# =============================================================================

def generate_orbital_positions(center, num, min_r, max_r, photo_size):
    positions = []
    orbit_spacing = max(photo_size) * 1.2
    num_orbits = max(2, int((max_r - min_r) / orbit_spacing))
    per_orbit = [num // num_orbits] * num_orbits
    for i in range(num % num_orbits):
        per_orbit[i] += 1
    
    for oi, count in enumerate(per_orbit):
        if count == 0:
            continue
        t = (oi + 0.5) / num_orbits
        radius = min_r + t * (max_r - min_r)
        start = random.uniform(0, 2 * math.pi)
        for i in range(count):
            angle = start + (2 * math.pi * i / count) + random.uniform(-0.1, 0.1)
            r = radius + random.uniform(-15, 15)
            positions.append((center[0] + int(r * math.cos(angle)), 
                            center[1] + int(r * math.sin(angle)), 
                            random.uniform(-20, 20)))
    return positions


def generate_spiral_positions(center, num, min_r, max_r, photo_size):
    positions = []
    angle = random.uniform(0, 2 * math.pi)
    radius = min_r
    r_inc = (max_r - min_r) / max(num, 1) * 0.8
    
    for _ in range(num):
        if radius > max_r:
            radius = min_r + random.uniform(0, (max_r - min_r) * 0.3)
            angle += math.pi / 2
        positions.append((center[0] + int(radius * math.cos(angle)),
                         center[1] + int(radius * math.sin(angle)),
                         random.uniform(-15, 15)))
        angle += math.pi / 3 + random.uniform(-0.2, 0.2)
        radius += r_inc + random.randint(-5, 10)
    return positions


def generate_cloud_positions(center, num, min_r, max_r, photo_size):
    positions = []
    for _ in range(num):
        r = min_r + (max_r - min_r) * (random.random() ** 0.7)
        angle = random.uniform(0, 2 * math.pi)
        positions.append((center[0] + int(r * math.cos(angle)),
                         center[1] + int(r * math.sin(angle)),
                         random.uniform(-25, 25)))
    return positions


def generate_brick_positions(center, num, min_r, max_r, photo_size, canvas_size, gap):
    brick_w, brick_h = photo_size
    cols = math.ceil(canvas_size[0] / (brick_w + gap)) + 2
    rows = math.ceil(canvas_size[1] / (brick_h + gap)) + 2
    start_x = (canvas_size[0] - cols * (brick_w + gap)) // 2
    start_y = (canvas_size[1] - rows * (brick_h + gap)) // 2
    
    all_pos = []
    for row in range(rows):
        offset = int((brick_w + gap) * 0.5) if row % 2 else 0
        for col in range(cols):
            x = start_x + col * (brick_w + gap) + offset + brick_w // 2
            y = start_y + row * (brick_h + gap) + brick_h // 2
            dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
            if min_r <= dist <= max_r and 0 <= x < canvas_size[0] and 0 <= y < canvas_size[1]:
                all_pos.append((x, y, 0, dist))
    
    all_pos.sort(key=lambda p: p[3])
    return [(x, y, r) for x, y, r, _ in all_pos[:num]]


# =============================================================================
# GÉNÉRATION DU NUAGE
# =============================================================================

def create_photo_cloud(main_img, photos, canvas_size, main_size, thumb_size,
                       layout, fade, fade_curve, gap, brick_ratio,
                       transparent, corner_radius, show_shadows, show_glow):
    
    bg_color = (0, 0, 0, 0) if transparent else (30, 30, 35, 255)
    canvas = Image.new('RGBA', canvas_size, bg_color)
    center = (canvas_size[0] // 2, canvas_size[1] // 2)
    max_distance = math.sqrt(center[0]**2 + center[1]**2)
    
    # Photo principale
    main_processed = main_img.copy()
    if main_processed.mode != 'RGBA':
        main_processed = main_processed.convert('RGBA')
    
    if layout == 'brick':
        main_processed = resize_cover(main_processed, main_size)
    else:
        main_processed = resize_to_fit(main_processed, main_size)
        if corner_radius > 0:
            main_processed = add_rounded_corners(main_processed, corner_radius + 5)
    
    main_processed = add_border(main_processed, 5, (255, 255, 255, 255))
    if corner_radius > 0 and layout != 'brick':
        main_processed = add_rounded_corners(main_processed, corner_radius + 8)
    
    # Rayons
    main_radius = max(main_processed.width, main_processed.height) // 2
    min_radius = main_radius + thumb_size // 2 + 20
    max_radius = min(canvas_size[0], canvas_size[1]) // 2 - thumb_size // 2
    
    if layout == 'brick':
        brick_h = int(thumb_size / brick_ratio)
        photo_size = (thumb_size, brick_h)
        min_radius = main_radius + 10
        max_radius = max(canvas_size) // 2 + max(photo_size)
    else:
        photo_size = (thumb_size, thumb_size)
    
    # Préparer les photos
    processed_photos = []
    for img in photos:
        img_copy = img.copy()
        if img_copy.mode != 'RGBA':
            img_copy = img_copy.convert('RGBA')
        
        if layout == 'brick':
            img_copy = resize_cover(img_copy, photo_size)
        else:
            img_copy = resize_to_fit(img_copy, photo_size)
            if corner_radius > 0:
                img_copy = add_rounded_corners(img_copy, corner_radius)
            img_copy = add_border(img_copy, 3, (255, 255, 255, 230))
            if corner_radius > 0:
                img_copy = add_rounded_corners(img_copy, corner_radius + 3)
        processed_photos.append(img_copy)
    
    if not processed_photos:
        return canvas
    
    # Positions
    num = len(processed_photos) if layout != 'brick' else len(processed_photos) * 10
    
    if layout == 'spiral':
        positions = generate_spiral_positions(center, num, min_radius, max_radius, photo_size)
    elif layout == 'orbital':
        positions = generate_orbital_positions(center, num, min_radius, max_radius, photo_size)
    elif layout == 'cloud':
        positions = generate_cloud_positions(center, num, min_radius, max_radius, photo_size)
    elif layout == 'brick':
        positions = generate_brick_positions(center, num, min_radius, max_radius, photo_size, canvas_size, gap)
    else:
        positions = generate_orbital_positions(center, num, min_radius, max_radius, photo_size)
    
    if layout != 'brick':
        random.shuffle(positions)
        positions.sort(key=lambda p: -math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2))
    
    # Dessiner
    for i, (x, y, rotation) in enumerate(positions):
        img = processed_photos[i % len(processed_photos)].copy()
        
        dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
        norm_dist = (dist / max_distance) ** fade_curve
        
        if fade > 0:
            brightness = 1.0 + norm_dist * fade + random.uniform(-0.05, 0.05)
            img = ImageEnhance.Brightness(img).enhance(brightness)
            sat = max(0.5, 1.0 - norm_dist * fade * 0.3)
            img = ImageEnhance.Color(img).enhance(sat)
        
        if layout != 'brick' and abs(rotation) > 0.5:
            img = img.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        if show_shadows and layout != 'brick':
            img = add_shadow(img, (5, 5), 10)
        
        canvas.paste(img, (x - img.width//2, y - img.height//2), img)
    
    # Lueur
    if show_glow and layout != 'brick':
        glow = Image.new('RGBA', (main_processed.width + 60, main_processed.height + 60), (0,0,0,0))
        gc = (glow.width // 2, glow.height // 2)
        for i in range(30, 0, -1):
            alpha = int(8 * (30 - i) / 30)
            ImageDraw.Draw(glow).ellipse([
                gc[0] - main_processed.width//2 - i, gc[1] - main_processed.height//2 - i,
                gc[0] + main_processed.width//2 + i, gc[1] + main_processed.height//2 + i
            ], fill=(255, 255, 255, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(15))
        canvas.paste(glow, (center[0] - glow.width//2, center[1] - glow.height//2), glow)
    
    # Photo principale
    if show_shadows:
        main_with_shadow = add_shadow(main_processed, (10, 10), 20)
    else:
        main_with_shadow = main_processed
    
    canvas.paste(main_with_shadow, 
                 (center[0] - main_with_shadow.width//2, center[1] - main_with_shadow.height//2),
                 main_with_shadow)
    
    return canvas


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Photo Cloud Generator", page_icon="📸", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .subtitle { text-align: center; color: #888; margin-bottom: 2em; }
    .format-info {
        background: linear-gradient(135deg, rgba(0,210,255,0.1), rgba(58,123,213,0.1));
        border: 1px solid rgba(0,210,255,0.2);
        border-radius: 10px;
        padding: 15px;
        font-size: 0.85em;
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-title">📸 Photo Cloud Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Créez des compositions artistiques avec vos photos • Tous formats supportés</p>', unsafe_allow_html=True)
    
    if 'photos' not in st.session_state:
        st.session_state.photos = []
    if 'main_index' not in st.session_state:
        st.session_state.main_index = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        st.subheader("🎨 Disposition")
        layout = st.radio("Mode", ['orbital', 'spiral', 'cloud', 'brick'],
                          format_func=lambda x: {'orbital': '🪐 Orbital', 'spiral': '🌀 Spirale', 
                                                  'cloud': '☁️ Nuage', 'brick': '🧱 Briques'}[x],
                          horizontal=True)
        
        st.divider()
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            canvas_width = st.number_input("Largeur", 800, 4096, 1920, 100)
        with col2:
            canvas_height = st.number_input("Hauteur", 600, 4096, 1080, 100)
        
        preset = st.selectbox("Preset", ["Personnalisé", "HD 1920×1080", "2K 2560×1440", "4K 3840×2160", 
                                          "Carré 1080", "Portrait 1080×1920", "Instagram 1080×1350"])
        presets = {"HD 1920×1080": (1920,1080), "2K 2560×1440": (2560,1440), "4K 3840×2160": (3840,2160),
                   "Carré 1080": (1080,1080), "Portrait 1080×1920": (1080,1920), "Instagram 1080×1350": (1080,1350)}
        if preset in presets:
            canvas_width, canvas_height = presets[preset]
        
        st.divider()
        st.subheader("🖼️ Photos")
        main_size = st.slider("Taille principale", 200, 800, 400, 50)
        thumb_size = st.slider("Taille miniatures", 80, 300, 150, 10)
        
        st.divider()
        st.subheader("✨ Effets")
        fade = st.slider("Éclaircissement", 0.0, 1.0, 0.5, 0.05)
        fade_curve = st.slider("Courbe fade", 0.3, 2.0, 1.0, 0.1)
        
        if layout == 'brick':
            st.divider()
            st.subheader("🧱 Briques")
            gap = st.slider("Espacement", 0, 15, 4)
            brick_ratio = st.slider("Ratio L/H", 1.0, 2.5, 1.5, 0.1)
        else:
            gap, brick_ratio = 4, 1.5
        
        st.divider()
        st.subheader("🎛️ Options")
        transparent = st.checkbox("Fond transparent", False)
        corner_radius = st.slider("Coins arrondis", 0, 30, 10) if layout != 'brick' else 0
        show_shadows = st.checkbox("Ombres", True)
        show_glow = st.checkbox("Lueur centrale", True) if layout != 'brick' else False
        
        st.divider()
        with st.expander("📋 Formats supportés"):
            st.markdown(get_supported_formats_info())
    
    # Main
    col_upload, col_preview = st.columns([1, 2])
    
    with col_upload:
        st.header("📁 Photos")
        
        all_ext = [ext.lstrip('.') for ext in get_all_supported_extensions()]
        uploaded_files = st.file_uploader("Glissez vos photos", type=all_ext, accept_multiple_files=True,
                                          help="JPEG, PNG, HEIC, RAW (CR2, NEF, ARW...), HDR...", key="uploader")
        
        if uploaded_files:
            current_names = {f.name for f in uploaded_files}
            stored_names = {p['name'] for p in st.session_state.photos}
            
            if current_names != stored_names:
                st.session_state.photos = []
                st.session_state.main_index = None
                
                progress = st.progress(0, "Chargement...")
                for i, f in enumerate(uploaded_files):
                    try:
                        img = load_image_auto(f)
                        st.session_state.photos.append({'name': f.name, 'image': img, 
                                                        'format': Path(f.name).suffix.lower()})
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                    progress.progress((i + 1) / len(uploaded_files))
                progress.empty()
        
        if st.session_state.photos:
            st.success(f"📷 {len(st.session_state.photos)} photo(s)")
            st.caption("👆 Cliquez pour définir la photo principale")
            
            cols = st.columns(3)
            for i, photo in enumerate(st.session_state.photos):
                with cols[i % 3]:
                    thumb = photo['image'].copy()
                    thumb.thumbnail((100, 100), Image.Resampling.LANCZOS)
                    if thumb.mode == 'RGBA':
                        bg = Image.new('RGB', thumb.size, (40, 40, 45))
                        bg.paste(thumb, mask=thumb.split()[3])
                        thumb = bg
                    
                    is_main = st.session_state.main_index == i
                    if st.button("⭐ PRINCIPALE" if is_main else "Sélectionner", key=f"btn_{i}",
                                 type="primary" if is_main else "secondary", width='stretch'):
                        st.session_state.main_index = i
                        st.rerun()
                    
                    fmt = photo['format'].upper().lstrip('.')
                    st.image(thumb, caption=f"{photo['name'][:10]}.. ({fmt})", width='stretch')
            
            st.divider()
            can_gen = st.session_state.main_index is not None and len(st.session_state.photos) >= 2
            
            if st.button("🎨 Générer", type="primary", disabled=not can_gen, width='stretch'):
                with st.spinner("Génération..."):
                    main_img = st.session_state.photos[st.session_state.main_index]['image']
                    others = [p['image'] for i, p in enumerate(st.session_state.photos) if i != st.session_state.main_index]
                    
                    st.session_state.result = create_photo_cloud(
                        main_img, others, (canvas_width, canvas_height), (main_size, main_size),
                        thumb_size, layout, fade, fade_curve, gap, brick_ratio,
                        transparent, corner_radius, show_shadows, show_glow)
                    st.rerun()
            
            if not can_gen:
                if st.session_state.main_index is None:
                    st.warning("⚠️ Sélectionnez une photo principale")
                elif len(st.session_state.photos) < 2:
                    st.warning("⚠️ Ajoutez au moins 2 photos")
        else:
            st.info("👆 Uploadez des photos")
            st.markdown("""<div class="format-info"><strong>📷 Formats:</strong> JPEG, PNG, GIF, WebP, TIFF, HEIC, 
            RAW (CR2, NEF, ARW, DNG...), HDR (EXR, HDR)</div>""", unsafe_allow_html=True)
    
    with col_preview:
        st.header("👁️ Prévisualisation")
        
        if st.session_state.result:
            result = st.session_state.result
            if transparent:
                st.caption("🔲 Fond transparent")
            st.image(result, width='stretch')
            
            buf_png = io.BytesIO()
            result.save(buf_png, format='PNG', optimize=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("💾 PNG", buf_png.getvalue(), f"photo_cloud_{layout}.png", "image/png", width='stretch')
            with c2:
                if not transparent:
                    buf_jpg = io.BytesIO()
                    result.convert('RGB').save(buf_jpg, format='JPEG', quality=95)
                    st.download_button("💾 JPEG", buf_jpg.getvalue(), f"photo_cloud_{layout}.jpg", "image/jpeg", width='stretch')
            
            st.caption(f"📐 {result.width}×{result.height} | {layout}")
        else:
            st.markdown("""<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:15px;
            padding:80px 40px;text-align:center;color:#666;"><div style="font-size:4em;margin-bottom:20px;">🖼️</div>
            <p>Uploadez des photos et cliquez sur "Générer"</p></div>""", unsafe_allow_html=True)
    
    st.divider()
    st.markdown('<p style="text-align:center;color:#666;font-size:0.85em;">Photo Cloud Generator • 🪐 Orbital · 🌀 Spirale · ☁️ Nuage · 🧱 Briques</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
