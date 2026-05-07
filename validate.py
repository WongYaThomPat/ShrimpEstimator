import os, cv2
from tools.tools2 import *
from prep.findratio import find_ratio
from scipy.optimize import linear_sum_assignment
from shrimp_test import get_model, predict_single

if __name__ == '__main__':
    os.system('cls')

    # PARAMS
    DEV = 'cpu'
    BASE_CH = 24
    MODEL_PATH = fr"models\shrimp_model_v2_{BASE_CH}.pth"
    V2 = True if 'v2' in MODEL_PATH else False
    CONF = 0.3
    MIN_DIST = 4
    UNSEEN = True
    VERBOSE = False
    SUBJECT_RANGE = [f'{i:03}' for i in range(50)]

    # METRICS
    all_shrimp_data = []
    file_summaries = []  
    total_tp, total_fp, total_fn = 0, 0, 0

    model = get_model(MODEL_PATH, BASE_CH, DEV, V2)
    for i, SUBJECT in enumerate(SUBJECT_RANGE):
        print(f'{i+1} / {len(SUBJECT_RANGE)}')

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

        if len(shrimp_list) > 0 and len(gt_shrimps) > 0:
            preds_centers = np.array([s['spine'] for s in shrimp_list])
            gts_centers = np.array([t['spine'] for t in gt_shrimps])

            cost_matrix = np.linalg.norm(preds_centers[:, np.newaxis] - gts_centers, axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_indices = []
            for p_idx, gt_idx in zip(row_ind, col_ind):
                dist = cost_matrix[p_idx, gt_idx]
                
                if dist <= 10: 
                    matched_indices.append((p_idx, gt_idx))
                    p_s = shrimp_list[p_idx]
                    gt_s = gt_shrimps[gt_idx]
                    
                    p_vol = np.pi * ((p_s['girth'] / 2)**2) * p_s['arc_length']
                    
                    all_shrimp_data.append({
                        'subject': SUBJECT,
                        'pred_len': p_s['arc_length'], 'gt_len': gt_s['arc_length'],
                        'pred_girth': p_s['girth'], 'gt_girth': gt_s['girth'],
                        'pred_vol': p_vol, 'gt_vol': gt_s['volume']
                    })

            tp = len(matched_indices)
            fp = len(shrimp_list) - tp
            fn = len(gt_shrimps) - tp
        else:
            tp, fp, fn = 0, len(shrimp_list), len(gt_shrimps)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        file_summaries.append({'subject': SUBJECT, 'tp': tp, 'fp': fp, 'fn': fn})

    # --- UNIVERSAL SUMMARY ---
    df_matches = pd.DataFrame(all_shrimp_data)
    
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)

    # 1. Detection Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"Detection: Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {2*(precision*recall)/(precision+recall+1e-9):.2f}")

    # 2. Size Metrics (MAPE & MedAPE)
    if not df_matches.empty:
        metrics = ['len', 'girth', 'vol']
        
        for m in metrics:
            df_matches[f'{m}_err'] = (df_matches[f'pred_{m}'] - df_matches[f'gt_{m}']) / df_matches[f'gt_{m}']
            df_matches[f'{m}_abs'] = df_matches[f'{m}_err'].abs()

        print(f"\n{'Metric':<10} | {'Bias (Mean)':>12} | {'MAPE (Mean)':>15} | {'MedAPE (Median)':>18}")
        print("-" * 65)
        
        for m in metrics:
            bias = df_matches[f'{m}_err'].mean() * 100
            mape = df_matches[f'{m}_abs'].mean() * 100
            medape = df_matches[f'{m}_abs'].median() * 100
            print(f"{m.capitalize():<10} | {bias:>11.2f}% | {mape:>14.2f}% | {medape:>17.2f}%")

        df_matches.to_csv(r'result\shrimp_detailed_results.csv', index=False)
        print("="*50)
        print("Full results saved to 'shrimp_detailed_results.csv'")
    else:
        print("No matches found to calculate size errors.")

   