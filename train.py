import os, torch
import numpy as np
from torch import nn
from pathlib import Path
from torchvision.transforms import v2, GaussianBlur
from ShrimpEstimator.tools.tools import ShrimpDataset, ShrimpModelV1, ShrimpModelV2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# NOTE: one more time and show loss profile

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    IM_PATH = Path(r'Shrimp\data\shrimp_ds\jpeg')
    HMP_PATH = Path(r'Shrimp\data\shrimp_ds\hm')
    IN_CH = 3
    BASE_CH = 32
    
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEV)
    
    BATCH_SIZE = 32
    EPOCHS = 180
    ROUND = 6
    LR = 1e-3
    SEED = 42

    # TRANSFORM = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Resize(size=(512, 512), antialias=True),
    #     v2.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2.0)),
    #     v2.RandomGrayscale(p=0.1),
    #     v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    # ])

    TRANSFORM = None

    X_all = sorted(list(IM_PATH.glob('*.jpg')))
    y_paths = sorted(list(HMP_PATH.glob('*.npy')))
    y_all = [np.load(p) for p in y_paths]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED
    )

    torch.manual_seed(SEED)
    train_ds = ShrimpDataset(X_train, y_train, TRANSFORM)
    test_ds = ShrimpDataset(X_test, y_test, TRANSFORM)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, pin_memory=True)

    model = ShrimpModelV2(in_ch=IN_CH, base_channels=BASE_CH).to(DEV)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.HuberLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=round(EPOCHS/ROUND), T_mult=1
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_test_loss = float('inf')
    save_path = Path(r"Shrimp\models")
    save_path.mkdir(exist_ok=True) 

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEV), y_batch.to(DEV)
            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        test_loss = 0
        with torch.inference_mode():
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(DEV), y_test_batch.to(DEV) 
                y_pred = model(X_test_batch)     
                loss = loss_fn(y_pred, y_test_batch)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"E{epoch+1:03d} | TRAIN: {avg_train_loss:.4f} | TEST: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({'model': model.state_dict()}, 
                       save_path / f"shrimp_model_v2_{BASE_CH}.pth")

        scheduler.step()
        # scheduler.step(avg_test_loss)