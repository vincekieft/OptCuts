import numpy as np

def _hsv_to_rgb_vec(h, s, v):
    """Vectorized HSV→RGB for arrays in [0,1]."""
    h = np.mod(h, 1.0)
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.where(i==0, v, np.where(i==1, q, np.where(i==2, p, np.where(i==3, p, np.where(i==4, t, v)))))
    g = np.where(i==0, t, np.where(i==1, v, np.where(i==2, v, np.where(i==3, q, np.where(i==4, p, p)))))
    b = np.where(i==0, p, np.where(i==1, p, np.where(i==2, t, np.where(i==3, v, np.where(i==4, v, q)))))
    return np.stack([r, g, b], axis=1)

def _default_palette(k, s=0.85, v=0.95, hue_offset=0.12):
    """K distinct bright colors in RGB [0,1]."""
    hues = (np.linspace(0, 1, k, endpoint=False) + hue_offset) % 1.0
    return _hsv_to_rgb_vec(hues, np.full(k, s), np.full(k, v))

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1/2.4)) - 0.055)

def label_probs_to_colors(P, palette=None, gamma_correct=False,
                       alpha_from_conf=False, return_uint8=True):
    """
    P: (F,K) probabilities per face per label.
    palette: (K,3) or (K,4) in [0,1]. If None, a distinct RGB palette is generated.
    gamma_correct: blend in linear light (recommended=True for nicer mixes).
    alpha_from_conf: set alpha from max probability per face (confidence).
    return_uint8: return RGBA uint8 (0..255) for trimesh.
    """
    P = np.asarray(P, dtype=np.float64)
    assert P.ndim == 2, "P must be (F,K)"
    F, K = P.shape

    # normalize rows (robust if they don't sum to 1)
    row_sum = P.sum(axis=1, keepdims=True)
    P = P / np.clip(row_sum, 1e-12, None)

    if palette is None:
        palette = _default_palette(K)  # (K,3)
    pal = np.asarray(palette, dtype=np.float64)
    if pal.shape[0] != K:
        raise ValueError(f"palette has {pal.shape[0]} colors, but P has K={K}")

    # split RGB / optional alpha
    if pal.shape[1] == 3:
        pal_rgb, pal_a = pal, None
    elif pal.shape[1] == 4:
        pal_rgb, pal_a = pal[:, :3], pal[:, 3:4]
    else:
        raise ValueError("palette must be (K,3) or (K,4)")

    # optional gamma-correct mixing
    rgb_basis = _srgb_to_linear(pal_rgb) if gamma_correct else pal_rgb
    blended_rgb = P @ rgb_basis
    blended_rgb = _linear_to_srgb(blended_rgb) if gamma_correct else blended_rgb
    blended_rgb = np.clip(blended_rgb, 0, 1)

    # alpha: palette-weighted or confidence-based or 1
    if alpha_from_conf:
        A = 0.3 + 0.7 * P.max(axis=1, keepdims=True)   # 0.3..1.0
    elif pal_a is not None:
        A = np.clip(P @ pal_a, 0, 1)
    else:
        A = np.ones((F, 1), dtype=np.float64)

    rgba = np.concatenate([blended_rgb, A], axis=1)
    if return_uint8:
        return (rgba * 255).round().astype(np.uint8)
    return rgba

def sdf_to_colors(nsdf_face: np.ndarray, alpha: int = 255) -> np.ndarray:
    t = np.clip(nsdf_face.astype(np.float64), 0.0, 1.0)
    pos  = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    colR = np.array([255, 240, 110,  60,  25], dtype=np.float64)
    colG = np.array([235, 220, 205, 170,  70], dtype=np.float64)
    colB = np.array([ 60, 100, 200, 230, 160], dtype=np.float64)
    r = np.interp(t, pos, colR)
    g = np.interp(t, pos, colG)
    b = np.interp(t, pos, colB)
    rgba = np.stack([r, g, b, np.full_like(r, alpha)], axis=1)
    return np.clip(rgba, 0, 255).astype(np.uint8)

def random_label_colors(k, alpha=255, seed=42):
    if seed is not None:
        np.random.seed(seed)
    rgb = np.random.randint(0, 256, size=(int(k), 3), dtype=np.uint8)
    return np.hstack([rgb, np.full((int(k), 1), alpha, dtype=np.uint8)])

def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    label_count = labels.max() + 1
    
    return random_label_colors(label_count, alpha=255, seed=42)[labels]
    