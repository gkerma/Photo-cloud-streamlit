#!/usr/bin/env python3
"""
Photo Cloud Generator - Application Streamlit
Créez des compositions artistiques de photos avec plusieurs modes de disposition.

Lancer avec: streamlit run photo_cloud_streamlit.py
"""

import streamlit as st
import math
import random
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# Configuration de la page
st.set_page_config(
    page_title="Photo Cloud Generator",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2em;
    }
    .layout-card {
        background: linear-gradient(135deg, rgba(0,210,255,0.1), rgba(58,123,213,0.1));
        border: 1px solid rgba(0,210,255,0.3);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .layout-card:hover {
        border-color: #00d2ff;
        transform: translateY(-2px);
    }
    .layout-card.selected {
        border-color: #00d2ff;
        background: rgba(0,210,255,0.2);
    }
    .layout-icon {
        font-size: 2em;
        margin-bottom: 5px;
    }
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 0.75em 1em;
        font-weight: bold;
    }
    .photo-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
        gap: 8px;
        padding: 10px;
        background: rgba(0,0,0,0.1);
        border-radius: 10px;
        max-height: 300px;
        overflow-y: auto;
    }
    .photo-item {
        aspect-ratio: 1;
        border-radius: 8px;
        overflow: hidden;
        border: 3px solid transparent;
        cursor: pointer;
        transition: all 0.2s;
    }
    .photo-item.selected {
        border-color: #4CAF50;
        box-shadow: 0 0 10px rgba(76,175,80,0.5);
    }
    .photo-item img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FONCTIONS DE TRAITEMENT D'IMAGE
# =============================================================================

def load_image(uploaded_file) -> Image.Image:
    """Charge une image uploadée."""
    img = Image.open(uploaded_file)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    return img


def resize_to_fit(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    """Redimensionne en conservant le ratio."""
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img


def resize_cover(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Redimensionne pour couvrir exactement (avec crop)."""
    target_w, target_h = target_size
    scale = max(target_w / img.width, target_h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def add_rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    """Ajoute des coins arrondis."""
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
    result = img.copy()
    result.putalpha(mask)
    return result


def add_border(img: Image.Image, width: int, color: tuple) -> Image.Image:
    """Ajoute une bordure."""
    if width <= 0:
        return img
    bordered = Image.new('RGBA', (img.width + width*2, img.height + width*2), color)
    bordered.paste(img, (width, width), img if img.mode == 'RGBA' else None)
    return bordered


def add_shadow(img: Image.Image, offset: tuple = (8, 8), blur: int = 15) -> Image.Image:
    """Ajoute une ombre portée."""
    padding = blur * 2 + max(abs(offset[0]), abs(offset[1]))
    shadow = Image.new('RGBA', (img.width + padding*2, img.height + padding*2), (0,0,0,0))
    shadow_shape = Image.new('RGBA', img.size, (0, 0, 0, 100))
    if img.mode == 'RGBA':
        shadow_shape.putalpha(img.split()[3])
    shadow.paste(shadow_shape, (padding + offset[0], padding + offset[1]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    shadow.paste(img, (padding, padding), img if img.mode == 'RGBA' else None)
    return shadow


# =============================================================================
# GÉNÉRATEURS DE POSITIONS
# =============================================================================

def generate_orbital_positions(center, num, min_r, max_r, photo_size):
    """Génère des positions sur orbites concentriques."""
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
            x = center[0] + int(r * math.cos(angle))
            y = center[1] + int(r * math.sin(angle))
            positions.append((x, y, random.uniform(-20, 20)))
    return positions


def generate_spiral_positions(center, num, min_r, max_r, photo_size):
    """Génère des positions en spirale."""
    positions = []
    angle = random.uniform(0, 2 * math.pi)
    radius = min_r
    r_inc = (max_r - min_r) / max(num, 1) * 0.8
    
    for _ in range(num):
        if radius > max_r:
            radius = min_r + random.uniform(0, (max_r - min_r) * 0.3)
            angle += math.pi / 2
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] + int(radius * math.sin(angle))
        positions.append((x, y, random.uniform(-15, 15)))
        angle += math.pi / 3 + random.uniform(-0.2, 0.2)
        radius += r_inc + random.randint(-5, 10)
    return positions


def generate_cloud_positions(center, num, min_r, max_r, photo_size):
    """Génère des positions aléatoires en nuage."""
    positions = []
    for _ in range(num):
        r = min_r + (max_r - min_r) * (random.random() ** 0.7)
        angle = random.uniform(0, 2 * math.pi)
        x = center[0] + int(r * math.cos(angle))
        y = center[1] + int(r * math.sin(angle))
        positions.append((x, y, random.uniform(-25, 25)))
    return positions


def generate_brick_positions(center, num, min_r, max_r, photo_size, canvas_size, gap):
    """Génère des positions en mur de briques."""
    positions = []
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
# FONCTION PRINCIPALE DE GÉNÉRATION
# =============================================================================

def create_photo_cloud(
    main_img: Image.Image,
    photos: list[Image.Image],
    canvas_size: tuple[int, int],
    main_size: tuple[int, int],
    thumb_size: int,
    layout: str,
    fade: float,
    fade_curve: float,
    gap: int,
    brick_ratio: float,
    transparent: bool,
    corner_radius: int,
    show_shadows: bool,
    show_glow: bool,
) -> Image.Image:
    """Génère le nuage de photos."""
    
    # Fond
    bg_color = (0, 0, 0, 0) if transparent else (30, 30, 35, 255)
    canvas = Image.new('RGBA', canvas_size, bg_color)
    center = (canvas_size[0] // 2, canvas_size[1] // 2)
    max_distance = math.sqrt(center[0]**2 + center[1]**2)
    
    # Photo principale
    main_processed = main_img.copy()
    if layout == 'brick':
        main_processed = resize_cover(main_processed, main_size)
    else:
        main_processed = resize_to_fit(main_processed, main_size)
        if corner_radius > 0:
            main_processed = add_rounded_corners(main_processed, corner_radius + 5)
    
    main_processed = add_border(main_processed, 5, (255, 255, 255, 255))
    if corner_radius > 0 and layout != 'brick':
        main_processed = add_rounded_corners(main_processed, corner_radius + 8)
    
    # Calcul des rayons
    main_radius = max(main_processed.width, main_processed.height) // 2
    min_radius = main_radius + thumb_size // 2 + 20
    max_radius = min(canvas_size[0], canvas_size[1]) // 2 - thumb_size // 2
    
    # Taille des photos selon le layout
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
    
    # Générer les positions
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
    
    # Trier par distance (loin = dessous)
    if layout != 'brick':
        random.shuffle(positions)
        positions.sort(key=lambda p: -math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2))
    
    # Dessiner les photos
    for i, (x, y, rotation) in enumerate(positions):
        img = processed_photos[i % len(processed_photos)].copy()
        
        # Éclaircissement selon la distance
        dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
        norm_dist = (dist / max_distance) ** fade_curve
        
        if fade > 0:
            brightness = 1.0 + norm_dist * fade + random.uniform(-0.05, 0.05)
            img = ImageEnhance.Brightness(img).enhance(brightness)
            sat = max(0.5, 1.0 - norm_dist * fade * 0.3)
            img = ImageEnhance.Color(img).enhance(sat)
        
        # Rotation
        if layout != 'brick' and abs(rotation) > 0.5:
            img = img.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Ombre
        if show_shadows and layout != 'brick':
            img = add_shadow(img, (5, 5), 10)
        
        # Coller
        canvas.paste(img, (x - img.width//2, y - img.height//2), img)
    
    # Lueur centrale
    if show_glow and layout != 'brick':
        glow = Image.new('RGBA', (main_processed.width + 60, main_processed.height + 60), (0,0,0,0))
        gc = (glow.width // 2, glow.height // 2)
        for i in range(30, 0, -1):
            alpha = int(8 * (30 - i) / 30)
            ImageDraw.Draw(glow).ellipse([
                gc[0] - main_processed.width//2 - i,
                gc[1] - main_processed.height//2 - i,
                gc[0] + main_processed.width//2 + i,
                gc[1] + main_processed.height//2 + i
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

def main():
    # Titre
    st.markdown('<h1 class="main-title">📸 Photo Cloud Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Créez des compositions artistiques avec vos photos</p>', unsafe_allow_html=True)
    
    # Initialiser l'état
    if 'photos' not in st.session_state:
        st.session_state.photos = []
    if 'main_index' not in st.session_state:
        st.session_state.main_index = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # Sidebar - Paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")
        
        # Layout
        st.subheader("🎨 Disposition")
        layout = st.radio(
            "Mode",
            options=['orbital', 'spiral', 'cloud', 'brick'],
            format_func=lambda x: {
                'orbital': '🪐 Orbital',
                'spiral': '🌀 Spirale',
                'cloud': '☁️ Nuage',
                'brick': '🧱 Briques'
            }[x],
            horizontal=True
        )
        
        st.divider()
        
        # Taille du canvas
        st.subheader("📐 Dimensions")
        col1, col2 = st.columns(2)
        with col1:
            canvas_width = st.number_input("Largeur", 800, 4096, 1920, 100)
        with col2:
            canvas_height = st.number_input("Hauteur", 600, 4096, 1080, 100)
        
        # Presets
        preset = st.selectbox("Preset", ["Personnalisé", "HD (1920×1080)", "2K (2560×1440)", "4K (3840×2160)", "Carré (1080×1080)", "Portrait (1080×1920)"])
        if preset == "HD (1920×1080)":
            canvas_width, canvas_height = 1920, 1080
        elif preset == "2K (2560×1440)":
            canvas_width, canvas_height = 2560, 1440
        elif preset == "4K (3840×2160)":
            canvas_width, canvas_height = 3840, 2160
        elif preset == "Carré (1080×1080)":
            canvas_width, canvas_height = 1080, 1080
        elif preset == "Portrait (1080×1920)":
            canvas_width, canvas_height = 1080, 1920
        
        st.divider()
        
        # Tailles des photos
        st.subheader("🖼️ Photos")
        main_size = st.slider("Taille photo principale", 200, 800, 400, 50)
        thumb_size = st.slider("Taille miniatures", 80, 300, 150, 10)
        
        st.divider()
        
        # Effets
        st.subheader("✨ Effets")
        fade = st.slider("Éclaircissement", 0.0, 1.0, 0.5, 0.05)
        fade_curve = st.slider("Courbe du fade", 0.3, 2.0, 1.0, 0.1)
        
        # Options brick
        if layout == 'brick':
            st.divider()
            st.subheader("🧱 Options Briques")
            gap = st.slider("Espacement (joints)", 0, 15, 4)
            brick_ratio = st.slider("Ratio L/H", 1.0, 2.5, 1.5, 0.1)
        else:
            gap = 4
            brick_ratio = 1.5
        
        st.divider()
        
        # Options générales
        st.subheader("🎛️ Options")
        transparent = st.checkbox("Fond transparent", False)
        corner_radius = st.slider("Coins arrondis", 0, 30, 10) if layout != 'brick' else 0
        show_shadows = st.checkbox("Ombres", True)
        show_glow = st.checkbox("Lueur centrale", True) if layout != 'brick' else False
    
    # Zone principale
    col_upload, col_preview = st.columns([1, 2])
    
    with col_upload:
        st.header("📁 Photos")
        
        # Upload
        uploaded_files = st.file_uploader(
            "Glissez vos photos ici",
            type=['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
            accept_multiple_files=True,
            key="uploader"
        )
        
        # Mettre à jour les photos
        if uploaded_files:
            st.session_state.photos = []
            for f in uploaded_files:
                try:
                    img = load_image(f)
                    st.session_state.photos.append({
                        'name': f.name,
                        'image': img
                    })
                except Exception as e:
                    st.error(f"Erreur: {f.name}: {e}")
        
        # Afficher les photos
        if st.session_state.photos:
            st.info(f"📷 {len(st.session_state.photos)} photo(s) chargée(s)")
            st.caption("👆 Cliquez sur une photo pour la définir comme principale")
            
            # Grille de sélection
            cols = st.columns(3)
            for i, photo in enumerate(st.session_state.photos):
                with cols[i % 3]:
                    # Créer une miniature
                    thumb = photo['image'].copy()
                    thumb.thumbnail((100, 100), Image.Resampling.LANCZOS)
                    
                    # Afficher avec bouton
                    is_main = st.session_state.main_index == i
                    
                    if st.button(
                        "⭐ PRINCIPALE" if is_main else "Sélectionner",
                        key=f"btn_{i}",
                        type="primary" if is_main else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.main_index = i
                        st.rerun()
                    
                    st.image(thumb, caption=photo['name'][:15], use_container_width=True)
            
            # Bouton de génération
            st.divider()
            
            can_generate = st.session_state.main_index is not None and len(st.session_state.photos) >= 2
            
            if st.button(
                "🎨 Générer le nuage",
                type="primary",
                disabled=not can_generate,
                use_container_width=True
            ):
                if can_generate:
                    with st.spinner("Génération en cours..."):
                        main_img = st.session_state.photos[st.session_state.main_index]['image']
                        other_photos = [
                            p['image'] for i, p in enumerate(st.session_state.photos)
                            if i != st.session_state.main_index
                        ]
                        
                        result = create_photo_cloud(
                            main_img=main_img,
                            photos=other_photos,
                            canvas_size=(canvas_width, canvas_height),
                            main_size=(main_size, main_size),
                            thumb_size=thumb_size,
                            layout=layout,
                            fade=fade,
                            fade_curve=fade_curve,
                            gap=gap,
                            brick_ratio=brick_ratio,
                            transparent=transparent,
                            corner_radius=corner_radius,
                            show_shadows=show_shadows,
                            show_glow=show_glow,
                        )
                        
                        st.session_state.result = result
                        st.rerun()
            
            if not can_generate:
                if st.session_state.main_index is None:
                    st.warning("⚠️ Sélectionnez une photo principale")
                elif len(st.session_state.photos) < 2:
                    st.warning("⚠️ Ajoutez au moins 2 photos")
        else:
            st.info("👆 Uploadez des photos pour commencer")
    
    with col_preview:
        st.header("👁️ Prévisualisation")
        
        if st.session_state.result is not None:
            # Afficher le résultat
            result = st.session_state.result
            
            # Fond damier pour la transparence
            if transparent:
                st.caption("🔲 Fond transparent (damier = zones transparentes)")
            
            st.image(result, use_container_width=True)
            
            # Bouton de téléchargement
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            buffer.seek(0)
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "💾 Télécharger PNG",
                    data=buffer.getvalue(),
                    file_name=f"photo_cloud_{layout}.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_dl2:
                # Version JPEG (sans transparence)
                if not transparent:
                    buffer_jpg = io.BytesIO()
                    result.convert('RGB').save(buffer_jpg, format='JPEG', quality=95)
                    buffer_jpg.seek(0)
                    st.download_button(
                        "💾 Télécharger JPEG",
                        data=buffer_jpg.getvalue(),
                        file_name=f"photo_cloud_{layout}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            # Infos
            st.caption(f"📐 {result.width} × {result.height} pixels | Mode: {layout}")
        else:
            # Placeholder
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 15px;
                padding: 80px 40px;
                text-align: center;
                color: #666;
            ">
                <div style="font-size: 4em; margin-bottom: 20px;">🖼️</div>
                <p>Uploadez des photos et cliquez sur "Générer"<br>pour créer votre composition</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85em;">
        <p>Photo Cloud Generator | 
        Layouts: 🪐 Orbital · 🌀 Spirale · ☁️ Nuage · 🧱 Briques</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
