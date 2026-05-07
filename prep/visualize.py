import numpy as np
import cv2
import os

def view_file(file_path):
    if not os.path.exists(file_path):
        print("File not found.")
        return

    hm = np.load(file_path)
    
    # Change (3, 224, 224) -> (224, 224, 3)
    vis = np.transpose(hm, (1, 2, 0))
    
    # Scale to 8-bit
    vis = (vis * 255).astype(np.uint8)
    
    # RGB to BGR for OpenCV
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    vis = cv2.resize(vis, (512, 512))
    
    cv2.imshow('Heatmap View', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Paste your specific filename here
    target = r'data\shrimp_ds\hm\shrimp_123.npy'
    view_file(target)