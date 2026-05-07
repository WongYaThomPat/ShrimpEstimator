import os
import numpy as np
import pandas as pd
import cv2
import ast

# TODO: Redo Environment

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    SUBJECT = "107"

    jpeg_path = fr"data\shrimp_ds\jpeg\shrimp_{SUBJECT}.jpg"
    csv_path = fr"data\shrimp_ds\csv\shrimp_{SUBJECT}.csv"

    jpeg = cv2.imread(jpeg_path)
    jpeg = cv2.resize(jpeg, dsize=(512, 512))

    csv = pd.read_csv(csv_path)

    if jpeg is not None:
        h, w, _ = jpeg.shape

        for kp_str in csv['Keypoints_XY']:
            kp = ast.literal_eval(kp_str)
            
            for key, pos in kp.items():
                _pos = (int(pos[0] * w), int((1 - pos[1]) * h))
                
                cv2.circle(jpeg, _pos, radius=1, color=(0, 0, 255), thickness=-1)
                cv2.putText(jpeg, key, _pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Radii definitions
        r1 = csv['Girth'] / 2          # Head (Full)
        r2 = r1 * 0.9                 # Tail (80%)
        
        # Applying your 2/3 and 1/3 weighted average
        r_weighted = (2/3 * r1) + (1/3 * r2)
        
        al = csv['ArcLength']
        vol_actual = csv['Volume']

        # Volume using the weighted radius
        vol_calc = np.round(np.pi * np.power(r_weighted, 2) * al, 4)
        vol_error = (vol_calc - vol_actual) / vol_actual * 100

        # --- NEW PRINT STATEMENTS ---
        print("-" * 30)
        print(f"Volume Analysis for Subject: {SUBJECT}")
        print("-" * 30)
        results_df = pd.DataFrame({
            'Actual Vol': vol_actual, 'Calc Vol': vol_calc, 'Error (%)': vol_error
        })
        print(results_df.to_string(index=False))
        print("-" * 30)
        # ----------------------------

        cv2.imshow('Shrimp Image', jpeg)
        
        cv2.waitKey(0)           
        cv2.destroyAllWindows()  

    else:
        print("Error: Could not read the image. Check your file path!")