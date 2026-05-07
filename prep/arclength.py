import os
import ast
import pandas as pd
import numpy as np
from prep.findratio import find_ratio
from scipy.integrate import quad

# NOTE: Straight Shrimps estimates are trash

def get_straight_len(kp_dict, ratio, a):
    head = np.array(kp_dict['head']) * a
    spine = np.array(kp_dict['spine']) * a
    tail = np.array(kp_dict['tail']) * a

    hs = np.linalg.norm(head - spine)
    st = np.linalg.norm(spine - tail)

    dist = np.power(1.006, hs) * hs + np.power(1.013, st) * st
    return dist * ratio

def get_poly_arc(kp_dict, ratio, a):
    pts = np.array([kp_dict['head'], kp_dict['spine'], kp_dict['tail']]) * a
    t = np.array([0, 1, 2])
    px = np.poly1d(np.polyfit(t, pts[:, 0], 2))
    py = np.poly1d(np.polyfit(t, pts[:, 1], 2))
    dpx, dpy = px.deriv(), py.deriv()
    length, _ = quad(lambda tv: np.sqrt(dpx(tv)**2 + dpy(tv)**2), 0, 2)
    return length * ratio

if __name__ == '__main__':
    os.system('cls')

    SUBJECT = '167'
    IMG_PATH = fr'data\shrimp_ds\jpeg\shrimp_{SUBJECT}.jpg'
    CSV_PATH = fr'data\shrimp_ds\csv\shrimp_{SUBJECT}.csv'
    
    a = 512

    RATIO = find_ratio(IMG_PATH, net_d= 100, a=a)
    df = pd.read_csv(CSV_PATH)

    for index, row in df.iterrows():
        kp = ast.literal_eval(row['Keypoints_XY'])
        if row['Pose'] == "Shrimp_Straight":
            calc_len = get_straight_len(kp, RATIO, a)
        else:
            calc_len = get_poly_arc(kp, RATIO, a)
        
        real_len = row['ArcLength']
        error = ((calc_len - real_len) / real_len) * 100

        print(f"Pose: {row['Pose']} | C: {calc_len:.4f} / R: {real_len:.4f} cm | Error: {error:.2f}%")
        