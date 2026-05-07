import os, cv2
from shrimp_test import get_model, predict_single
from prep.findratio import find_ratio
from tools.tools2 import *

if __name__ == '__main__':
    os.system('cls')

    SUBJECT = '002'
    DEV = 'cpu'
    BASE_CH = 32
    MODEL_PATH = fr"models\shrimp_model_v2_{BASE_CH}.pth"
    CONF = 0.3
    MIN_DIST = 4
    UNSEEN = True
    VERBOSE = False
    V2 = True if 'v2' in MODEL_PATH else False

    model = get_model(MODEL_PATH, BASE_CH, DEV, V2)

    heatmap = predict_single(model, SUBJECT, DEV, UNSEEN)
    if UNSEEN:
        og_path = fr'data\validate_set\jpeg\shrimp_{SUBJECT}.jpg'
    else:
        og_path = fr'data\shrimp_ds\jpeg\shrimp_{SUBJECT}.jpg'

    og_img = cv2.imread(og_path, 1)
    og_img = cv2.resize(og_img, (512, 512))

    UPPX = find_ratio(og_path, a=og_img.shape[1])
    part_coords = get_part_coords(heatmap, CONF, MIN_DIST, VERBOSE)

    shrimp_list = group_parts(part_coords, og_img.shape[1])

    pred_shrimps = estimate_length(shrimp_list, UPPX)
    pred_shrimps = estimate_girth(pred_shrimps, og_img, UPPX)

    gt_shrimps = load_ground_truth(SUBJECT, UNSEEN, og_img.shape[1])
    plot_comparison(pred_shrimps, gt_shrimps, og_img)
