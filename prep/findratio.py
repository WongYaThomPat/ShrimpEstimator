import os
import cv2
import numpy as np

def find_ratio(img_path : str, net_d : float = 100, a : int = 512):
    _img = cv2.imread(img_path)
    if _img is None:
        return
        
    _img = cv2.resize(_img, (a, a))
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))

    window_size = 3
    density_map = cv2.boxFilter(mag, -1, (window_size, window_size), normalize=False)

    _, mask = cv2.threshold(density_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # _, _, _, h = cv2.boundingRect(mask)

    area = cv2.countNonZero(mask)
    h = np.sqrt(area)

    return net_d / h


if __name__ == '__main__':
    os.system('cls')
    NET_D = 100
    path = r'data\shrimp_ds\jpeg\shrimp_014.jpg'
    
    print(find_ratio(path, NET_D))