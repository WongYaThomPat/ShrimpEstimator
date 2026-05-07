import numpy as np, pandas as pd
from numpy.polynomial import Polynomial
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.integrate import quad
import matplotlib.pyplot as plt
import ast, cv2

def get_part_coords(heatmap: np.ndarray, alpha: float = 0.2, min_dist: int = 5, verbose : bool = False) -> dict:
    parts = ['head', 'spine', 'tail']
    part_coords = {part: {'coords': [], 'cnt': 0} for part in parts}
    
    _, h, w = heatmap.shape

    for i, part_name in enumerate(parts):
        if i >= heatmap.shape[0]:
            break
            
        channel = heatmap[i]
        peaks = peak_local_max(channel, min_distance=min_dist, threshold_abs=alpha)

        for py, px in peaks:
            part_coords[part_name]['coords'].append((px / w, py / h))

        part_coords[part_name]['cnt'] = len(peaks)
    
    if verbose:
        for part in parts:
            print(f'{part} : {part_coords[part_name]['cnt']}')

    return part_coords


def group_parts(part_coords : dict, d : int = 512) -> list:
    shrimp_list = []
    max_dist = int(d * 0.2)

    h_coords = np.array(part_coords['head']['coords']) * d
    s_coords = np.array(part_coords['spine']['coords']) * d
    t_coords = np.array(part_coords['tail']['coords']) * d

    dist_hs = cdist(h_coords, s_coords)
    dist_ts = cdist(t_coords, s_coords)

    used_heads = set()
    used_tails = set()
    used_spines = set()

    hs_pairs = []
    for h_idx in range(len(h_coords)):
        for s_idx in range(len(s_coords)):
            if dist_hs[h_idx, s_idx] < max_dist:
                hs_pairs.append((dist_hs[h_idx, s_idx], h_idx, s_idx))
    
    hs_pairs.sort()

    spine_to_head = {}
    for _, h_i, s_i in hs_pairs:
        if h_i not in used_heads and s_i not in used_spines:
            spine_to_head[s_i] = h_coords[h_i]
            used_heads.add(h_i)
            used_spines.add(s_i)

    ts_pairs = []
    for t_idx in range(len(t_coords)):
        for s_idx in range(len(s_coords)):
            if dist_ts[t_idx, s_idx] < max_dist:
                ts_pairs.append((dist_ts[t_idx, s_idx], t_idx, s_idx))
    
    ts_pairs.sort()
    
    spine_to_tail = {}
    used_spines_for_tails = set() 
    for _, t_i, s_i in ts_pairs:
        if t_i not in used_tails and s_i not in used_spines_for_tails:
            spine_to_tail[s_i] = t_coords[t_i]
            used_tails.add(t_i)
            used_spines_for_tails.add(s_i)

    for s_idx in range(len(s_coords)):
        head = spine_to_head.get(s_idx, None)
        spine = s_coords[s_idx]
        tail = spine_to_tail.get(s_idx, None)

        if head is not None and tail is not None:
            v1 = head - spine
            v2 = tail - spine
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                dot_prod = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
                
                if angle <= 70:
                    head = None
                    tail = None

        shrimp_list.append({
            'head': head,
            'spine': spine,
            'tail': tail
        })

    return shrimp_list


def load_ground_truth(subject: str, unseen: bool, d: int = 512):
    path = fr'Shrimp\data\validate_set\csv\shrimp_{subject}.csv' if unseen else \
           fr'Shrimp\data\shrimp_ds\csv\shrimp_{subject}.csv'
    
    df = pd.read_csv(path)
    gt_list = []
    
    for _, row in df.iterrows():
        kp = ast.literal_eval(row['Keypoints_XY'])
        shrimp = {
            'head': (kp['head'][0] * d, (1 - kp['head'][1]) * d) if kp['head'] else None,
            'spine': (kp['spine'][0] * d, (1 - kp['spine'][1]) * d) if kp['spine'] else None,
            'tail': (kp['tail'][0] * d, (1 - kp['tail'][1]) * d) if kp['tail'] else None,
            'arc_length': row['ArcLength'],
            'girth': row['Girth'],
            'volume': row['Volume']
        }
        gt_list.append(shrimp)

    return gt_list


def plot_comparison(pred_list: list, gt_list: list, og_img: np.ndarray):
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    img_rgb = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
    
    titles = ["Model Predictions", "Ground Truth"]
    data_sources = [pred_list, gt_list]

    for ax, shrimp_list, title in zip(axes, data_sources, titles):
        ax.imshow(img_rgb)
        
        num_shrimp = len(shrimp_list)
        cmap = plt.get_cmap('jet')
        colors = [cmap(i) for i in np.linspace(0, 1, num_shrimp)] if num_shrimp > 0 else []

        for i, shrimp in enumerate(shrimp_list):
            color = colors[i]
            h, s, t_pt = shrimp.get('head'), shrimp.get('spine'), shrimp.get('tail')
            
            if all(pt is not None for pt in [h, s, t_pt]):
                pts = np.array([h, s, t_pt])
                d1 = np.linalg.norm(pts[1] - pts[0])
                d2 = np.linalg.norm(pts[2] - pts[1])
                t_steps = np.array([0, d1, d1 + d2])
                
                if (d1 + d2) > 0:
                    poly_x = Polynomial.fit(t_steps, pts[:, 0], 2)
                    poly_y = Polynomial.fit(t_steps, pts[:, 1], 2)
                    t_fine = np.linspace(0, d1 + d2, 50)
                    ax.plot(poly_x(t_fine), poly_y(t_fine), color=color, linewidth=2, alpha=0.8)
            else:
                valid_pts = np.array([p for p in [h, s, t_pt] if p is not None])
                if len(valid_pts) > 1:
                    ax.plot(valid_pts[:, 0], valid_pts[:, 1], color=color, linewidth=2, alpha=0.8, linestyle='--')

            for pt in [h, s, t_pt]:
                if pt is not None:
                    ax.scatter(pt[0], pt[1], color=color, s=40, edgecolors='white', linewidths=1, zorder=5)

            if s is not None:
                sx, sy = s
                length = shrimp.get('arc_length', 0)
                girth = shrimp.get('girth', 0)
                
                if title == "Model Predictions":
                    l_min, l_max = length * 0.95, length * 1.05
                    g_min, g_max = girth * 0.95, girth * 1.05
                    label = f"L: {l_min:.1f}-{l_max:.1f}\nD: {g_min:.1f}-{g_max:.1f}"
                else:
                    label = f"L: {length:.1f}\nD: {girth:.1f}"
                
                ax.text(sx + 5, sy - 5, label, color='white', fontsize=7, fontweight='bold',
                        bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2), zorder=10)

        ax.set_title(f"{title} (n={len(shrimp_list)})", fontsize=15)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def estimate_length(shrimp_list: list, ratio: float):
    for shrimp in shrimp_list:
        valid_points = {k: v for k, v in shrimp.items() if v is not None}
        num_none = 3 - len(valid_points)
        al_px = 0 

        if num_none == 1:
            if shrimp['head'] is None:
                al_px = 1.8 * np.linalg.norm(np.array(shrimp['spine']) - np.array(shrimp['tail']))
            elif shrimp['tail'] is None:
                al_px = 1.8 * np.linalg.norm(np.array(shrimp['spine']) - np.array(shrimp['head']))
            elif shrimp['spine'] is None:
                al_px = np.linalg.norm(np.array(shrimp['head']) - np.array(shrimp['tail']))
        
        elif num_none == 0:
            pts = np.array([shrimp['head'], shrimp['spine'], shrimp['tail']])
            t = np.array([0, 1, 2])
            poly_x = Polynomial.fit(t, pts[:, 0], 2)
            poly_y = Polynomial.fit(t, pts[:, 1], 2)
            dx, dy = poly_x.deriv(), poly_y.deriv()
            al_px, _ = quad(lambda tv: np.sqrt(dx(tv)**2 + dy(tv)**2), 0, 2)

        shrimp['arc_length_px'] = al_px 
        shrimp['arc_length'] = al_px * ratio

    return shrimp_list


def estimate_girth(shrimp_list: list, og_img: np.ndarray, ratio: float):
    def get_girth_from_crop(spine_coord, img):
        d = img.shape[1]
        c = int(d * 0.03) 
        x, y = int(spine_coord[0]), int(spine_coord[1])
        x_min, x_max = max(0, x - c), min(x + c, d - 1)
        y_min, y_max = max(0, y - c), min(y + c, d - 1)
        
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0: return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        h, w = dist_map.shape
        radius = dist_map[h // 2, w // 2]
        return float(radius * 2)

    for shrimp in shrimp_list:
        measured_girth_px = 0
        has_parts = all(shrimp.get(part) is not None for part in ['head', 'spine', 'tail'])

        if has_parts:
            measured_girth_px = get_girth_from_crop(shrimp['spine'], og_img)
        
        if measured_girth_px < 3:
            girth_px = shrimp['arc_length_px'] * 0.22
        else:
            girth_px = measured_girth_px

        girth_phys = girth_px * ratio
        shrimp['girth'] = girth_phys
    
    return shrimp_list