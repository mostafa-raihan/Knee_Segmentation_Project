import os
from args import get_args
import torch
import torch.nn as nn
from utils import dice_loss_from_logits, dice_score_from_logits


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)

            optimizer.zero_grad()
            output = model(images)

            loss_bce = bce(output, masks)
            loss_dice = dice_loss_from_logits(output, masks)
            loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = validate_model(model, val_loader, bce, device)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Dice {val_dice:.4f}")

        # save the best model (lowest validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            print("   -> best model saved")


def validate_model(model, val_loader, bce, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)
            output = model(images)

            loss_bce = bce(output, masks)
            loss_dice = dice_loss_from_logits(output, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            running_dice += dice_score_from_logits(output, masks)

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)

    return val_loss, val_dice