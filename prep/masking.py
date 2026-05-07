import os
from pathlib import Path
import cv2, numpy as np
import pandas as pd, ast

def draw_gaussian(heatmap, center, sigma, k_size):
    tmp_size = k_size // 2
    x, y = center
    mu = tmp_size
    grid_x = np.arange(k_size)
    grid_y = np.arange(k_size)[:, None]
    v = np.exp(-((grid_x - mu)**2 + (grid_y - mu)**2) / (2 * sigma**2))
    y_min, y_max = max(0, y - tmp_size), min(heatmap.shape[0], y + tmp_size + 1)
    x_min, x_max = max(0, x - tmp_size), min(heatmap.shape[1], x + tmp_size + 1)
    ky_min, ky_max = tmp_size - (y - y_min), tmp_size + (y_max - y)
    kx_min, kx_max = tmp_size - (x - x_min), tmp_size + (x_max - x)
    heatmap[y_min:y_max, x_min:x_max] = np.maximum(
        heatmap[y_min:y_max, x_min:x_max], v[ky_min:ky_max, kx_min:kx_max]
    )

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    BASE_PATH = r'Shrimp\data\shrimp_ds'
    CSV_PATH = Path(os.path.join(BASE_PATH, 'csv'))
    HM_PATH = Path(os.path.join(BASE_PATH, 'hm'))
    HM_PATH.mkdir(parents=True, exist_ok=True)

    d = 128
    if not CSV_PATH.exists(): exit()

    all_girths = []
    csv_files = list(CSV_PATH.glob('*.csv'))

    for csv in csv_files:
        df = pd.read_csv(csv)
        all_girths.extend(df['Girth'].tolist())
    
    b33 = np.quantile(all_girths, 0.33)
    g_min = min(all_girths)
    g_max = max(all_girths)
    s_min, s_max = 0.6, 0.8

    for csv in csv_files:
        hm = np.zeros((3, d, d), dtype=np.float32)
        df = pd.read_csv(csv)
        
        for _, row in df.iterrows():
            kp = ast.literal_eval(row['Keypoints_XY'])
            g = row['Girth']
            

            k_size = 3 if g < b33 else 5
            sigma = s_min + (g - g_min) * (s_max - s_min) / (g_max - g_min + 1e-6)

            parts = ['head', 'spine', 'tail']
            for i in range(len(parts)):
                pos = kp[parts[i]]
                x_c = np.clip(int(pos[0] * d), 0, d - 1)
                y_c = np.clip(int((1 - pos[1]) * d), 0, d - 1)
                
                if i == 2 :
                    sigma *= 0.8

                draw_gaussian(hm[i], (x_c, y_c), sigma, k_size)

        hm = np.clip(hm, 0, 1.0)
        np.save(str(HM_PATH / csv.name.replace('.csv', '.npy')), hm)