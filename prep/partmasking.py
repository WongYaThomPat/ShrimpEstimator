import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

os.system('cls' if os.name == 'nt' else 'clear')

# --- CONFIG ---
img_folder = r'data\feature_bundle\og'
output_folder = r'data\feature_bundle\output_masks'
SIZE = (256, 256)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# --------------

for root, dirs, files in os.walk(img_folder):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Path logic
            img_path = os.path.join(root, file)
            
            # Extract names for: {body_part}_{perspective}
            # root typically looks like: ...\og\head
            body_part = os.path.basename(root) 
            perspective = os.path.splitext(file)[0]
            new_name = f"{body_part}_{perspective}.png"

            # Processing
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, SIZE)

            # Masking Background
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            bg_intensity = np.argmax(hist)
            rng = 0.03
            mask = (img >= bg_intensity*(1-rng)) & (img <= bg_intensity*(1+rng))
            img_cleaned = img.copy()
            img_cleaned[mask] = 0

            # Outline & Sobel
            _, th = cv2.threshold(img_cleaned, 10, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
            morph_outline = cv2.morphologyEx(th_closed, cv2.MORPH_GRADIENT, kernel)

            blurred = cv2.GaussianBlur(img_cleaned, (3, 3), 0)
            sobelx = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
            sobely = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
            sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

            # Final Product
            combined_edges = cv2.addWeighted(morph_outline, 0.5, sobel_combined, 0.5, 0)
            combined_edges = cv2.normalize(combined_edges, None, 0, 255, cv2.NORM_MINMAX)

            # Save result
            save_path = os.path.join(output_folder, new_name)
            cv2.imwrite(save_path, combined_edges)
            print(f"Processed: {new_name}")

print("\nAll done! Check the 'output_masks' folder.")