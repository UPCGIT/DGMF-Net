import argparse
import os
import torch
from utils.dataset import make_dataloader
from train import train, validation
from model.model import Proposed
from Loss import FocalLoss

parser = argparse.ArgumentParser(description="multi_modal RS classification")
parser.add_argument("--model_name", type=str, default='proposed')
parser.add_argument("--dataset_name", type=str, default="YRD-MVD")
parser.add_argument("--dataset_dir", type=str, default='')
parser.add_argument("--use_pca", type=bool, default=False)
parser.add_argument("--pca_component", type=int, default='')
parser.add_argument("--patch_size", type=int, default='')
parser.add_argument("--epoch", type=int, default='')
parser.add_argument("--warmup_epochs", type=int, default='')
parser.add_argument("--batch_size", type=int, default='')
parser.add_argument("--lr", type=float, default='')
parser.add_argument("--weight_decay", type=float, default='')
parser.add_argument("--use_self_scheduler", type=bool, default=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--saving_path", type=str, default='')
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--is_train", type=bool, default=True)
parser.add_argument("--test_freq", type=int, default='')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader, whole_loader, gt= make_dataloader(args)

model = Proposed(args.dataset_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

criterion = FocalLoss()

if args.is_train:

    train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler)

else:
    model_path = os.path.join(args.saving_path,args.dataset_name, 'model_best.pth')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    test_loss, results, test_time =validation(model, criterion, test_loader, device)
    cm = results['Confusion matrix']
    pa = results['PA']
    aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
    save_txt_path = os.path.join(args.saving_path, args.dataset_name, f"test_patch{args.lr}.txt")
    with open(save_txt_path, 'a', encoding='utf-8') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"OA: {oa:.4f}\n")
        f.write(f"AA: {aa:.4f}\n")
        f.write(f"Kappa: {kappa:.4f}\n\n")

        f.write("Per-class accuracy (PA):\n")
        for i, acc in enumerate(pa):
            f.write(f"Class {i}: {acc:.4f}\n")

        f.write("\nConfusion Matrix:\n")
        for row in cm:
            f.write(" ".join([str(int(x)) for x in row]) + "\n")






