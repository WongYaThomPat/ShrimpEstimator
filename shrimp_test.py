import os
import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from tools.tools import ShrimpModelV1, ShrimpModelV2, ShrimpDataset

def get_model(model_path, base_ch, device, v2=False):
    if not v2:
        model = ShrimpModelV1(base_channels=base_ch).to(device)
    else:
        model = ShrimpModelV2(base_channels=base_ch).to(device)
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def predict_single(model, subject: str, device: str, validate_set : bool = False):
    if not validate_set:
        X_path = fr'data\shrimp_ds\jpeg\shrimp_{subject}.jpg'
    else:
        X_path = fr'data\validate_set\jpeg\shrimp_{subject}.jpg'
    
    dummy_y = [np.zeros((3, 512, 512), dtype=np.float32)]
    test_ds = ShrimpDataset(x=[X_path], y=dummy_y)
    X, _ = test_ds[0]
    X = X.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(X)
        mask = outputs[0].cpu().numpy()
    return mask

def plot_to_ax(ax, heatmap: np.ndarray, subject: str,  alpha: float = 0.8):
    img_path = fr'data\shrimp_ds\jpeg\shrimp_{subject}.jpg'
    img = cv2.imread(img_path, 1)
    if img is None: return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for i in range(min(heatmap.shape[0], 3)):
        channel = heatmap[i]
        channel_resized = cv2.resize(channel, (w, h), interpolation=cv2.INTER_LINEAR)
        channel_resized = np.clip(channel_resized, 0, 1)
        
        for c in range(3):
            overlay[..., c] = np.maximum(overlay[..., c], channel_resized * colors[i][c])
        overlay[..., 3] = np.maximum(overlay[..., 3], channel_resized * alpha)

    ax.imshow(img)
    ax.imshow(overlay, interpolation='bilinear')
    ax.set_title(f"ID: {subject}")
    ax.axis('off')

if __name__ == '__main__':
    os.system('cls')

    DEV = 'cpu'
    BASE_CH = 32
    MODEL_PATH = fr"models\best_shrimp_model_{BASE_CH}.pth"
    
    model = get_model(MODEL_PATH, BASE_CH, DEV)
    subjects = [f'{np.random.randint(0, 599):03}' for _ in range(6)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, subj in enumerate(subjects):
        heatmap = predict_single(model, subj, DEV)
        plot_to_ax(axes[i], heatmap, subj)
        
    plt.tight_layout()
    plt.show()