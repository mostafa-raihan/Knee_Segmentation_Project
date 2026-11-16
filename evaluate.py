import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from args import get_args
from dataset import KneeDataset, read_xray
from torch.utils.data import DataLoader
from model import UNetLext
from utils import dice_score_from_logits


#1: Arguments
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#2: Load the BEST model saved during training
model = UNetLext(
    input_channels=1,
    output_channels=1,
    pretrained=False,
    path_weights='',
    path_pretrained='',
    restore_weights=False,
)
model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pth'), map_location=device))
model = model.to(device)
model.eval()
print("Best model loaded from:", os.path.join(args.out_dir, 'best_model.pth'))

#3: Performance metric -> Dice score on the validation set (has true masks)
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))
val_dataset = KneeDataset(val_set)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

dice_scores = []
with torch.no_grad():
    for data_batch in val_loader:
        images = data_batch['xray'].to(device)
        masks = data_batch['mask'].to(device)
        output = model(images)
        dice_scores.append(dice_score_from_logits(output, masks))

dice_series = pd.Series(dice_scores)
print(f"Validation Dice -> mean: {dice_series.mean():.4f} | "
      f"min: {dice_series.min():.4f} | max: {dice_series.max():.4f}")

#4: Visualization -> validation examples (x-ray | true mask | predicted mask)
plt.figure(figsize=(12, 12))
with torch.no_grad():
    for i, data_batch in enumerate(val_loader):
        if i == 3:
            break
        images = data_batch['xray'].to(device)
        output = model(images)
        pred = (torch.sigmoid(output) > 0.5).float()

        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[0, 0].cpu().numpy(), cmap='gray')
        plt.title("X-ray")
        plt.axis('off')

        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(data_batch['mask'][0, 0].numpy(), cmap='gray')
        plt.title("True mask")
        plt.axis('off')

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(pred[0, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Predicted (Dice {dice_scores[i]:.2f})")
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, 'val_predictions.png'))
plt.show()

#5: Test set -> predicted masks (test images have NO true mask)
test_set = pd.read_csv(os.path.join(args.csv_dir, 'test.csv'))

plt.figure(figsize=(8, 16))
with torch.no_grad():
    for i in range(4):
        xray = read_xray(test_set['xrays'].iloc[i])
        images = torch.from_numpy(xray).unsqueeze(0).to(device)
        output = model(images)
        pred = (torch.sigmoid(output) > 0.5).float()

        plt.subplot(4, 2, i * 2 + 1)
        plt.imshow(xray[0], cmap='gray')
        plt.title("Test X-ray")
        plt.axis('off')

        plt.subplot(4, 2, i * 2 + 2)
        plt.imshow(xray[0], cmap='gray')
        plt.imshow(pred[0, 0].cpu().numpy(), cmap='Reds', alpha=0.4)
        plt.title("Predicted mask (overlay)")
        plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, 'test_predictions.png'))
plt.show()