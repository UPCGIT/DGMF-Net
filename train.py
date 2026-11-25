import os
import math
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
import time

def train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler=None):
    print("=================================== Training ===================================")
    total_train_start = time.time()
    best_acc = 0.0
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        model.train()
        losses = AverageMeter()
        tar_list, pre_list = [], []
        for batch_idx, (hsi, lidar, batch_target) in enumerate(train_loader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            batch_target = batch_target.to(device)
            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), batch_target.size(0))
            batch_pred = batch_out.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
            batch_tar = batch_target.detach().cpu().numpy().astype(np.int64)
            pre_list.append(batch_pred)
            tar_list.append(batch_tar)
        pre = np.concatenate(pre_list) if len(pre_list) else np.array([], dtype=np.int64)
        tar = np.concatenate(tar_list) if len(tar_list) else np.array([], dtype=np.int64)
        train_loss = losses.avg
        train_results = compute_metrics(pre, tar)
        train_oa = train_results['OA']
        if args.use_self_scheduler:
            lr = step_learning_rate(args, epoch, optimizer)
        else:
            if scheduler is not None:
                scheduler.step()

        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch:03d}/{args.epoch}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Train OA: {train_oa * 100:.2f}% | "
              f"Epoch Time: {epoch_time:.2f}s")
        # ===== 验证 =====
        if (epoch % args.test_freq == 0) or (epoch == args.epoch):
            test_loss, results, test_time = validation(model, criterion, test_loader, device)
            aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
            is_best = oa >= best_acc
            best_acc = max(oa, best_acc)
            save_checkpoint(model, is_best, args.saving_path, args.dataset_name, epoch=epoch, acc=best_acc)
            print(f"Test | "
                  f"Loss: {test_loss:.4f} | "
                  f"OA: {oa * 100:.2f}% | "
                  f"AA: {aa * 100:.2f}% | "
                  f"Kappa: {kappa:.4f} | "
                  f"Test Time: {test_time:.2f}s")
    total_train_time = time.time() - total_train_start
    print("=================================== Training Finished ===================================")
    print(f">>> Total Training Time: {total_train_time:.2f} s")
def validation(model, criterion, test_loader, device):
    model.eval()
    start_time = time.time()
    losses = AverageMeter()
    tar_list, pre_list = [], []
    with torch.no_grad():
        for batch_idx, (hsi, lidar, batch_target) in enumerate(test_loader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            batch_target = batch_target.to(device)
            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)
            losses.update(loss.item(), batch_target.size(0))
            batch_pred = batch_out.detach().argmax(dim=1).cpu().numpy().astype(np.int64)
            batch_tar = batch_target.detach().cpu().numpy().astype(np.int64)
            pre_list.append(batch_pred)
            tar_list.append(batch_tar)
    pre = np.concatenate(pre_list) if len(pre_list) else np.array([], dtype=np.int64)
    tar = np.concatenate(tar_list) if len(tar_list) else np.array([], dtype=np.int64)
    total_loss = losses.avg
    results = compute_metrics(pre, tar)
    test_time = time.time() - start_time
    return total_loss, results, test_time

def save_checkpoint(network, is_best, saving_path, dataset_name, **kwargs):
    save_dir = os.path.join(saving_path, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        tqdm.write("epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(save_dir, 'model_best.pth'))
class AverageMeter(object):
    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)
        self.avg = self.sum / self.cnt if self.cnt > 0 else 0.0

def compute_metrics(pred, target):
    """Compute metrics: Confusion Matrix, OA, PA, AA, Kappa"""
    results = {}
    cm = confusion_matrix(target, pred)
    results['Confusion matrix'] = cm
    total = np.sum(cm)
    oa = float(np.trace(cm)) / float(total) if total > 0 else 0.0
    results['OA'] = oa
    n_classes = cm.shape[0]
    pa = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        den = np.sum(cm[i, :])
        pa[i] = (cm[i, i] / den) if den > 0 else 0.0
    results['PA'] = pa
    aa = float(np.mean(pa)) if n_classes > 0 else 0.0
    results['AA'] = aa
    # kappa
    if total > 0:
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
        kappa = (oa - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    else:
        kappa = 0.0
    results['Kappa'] = kappa
    return results

def step_learning_rate(args, epoch, optimizer):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs

    if epoch <= warm_epochs:
        lr_adj = epoch / warm_epochs
    elif epoch < int(0.4 * total_epochs):
        lr_adj = 1.0
    elif epoch < int(0.7 * total_epochs):
        lr_adj = 0.1
    elif epoch < int(1 * total_epochs):
        lr_adj = 0.01
    else:
        lr_adj = 0.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj

def cosine_learning_rate(args, epoch, optimizer):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs

    if epoch <= warm_epochs:
        lr_adj = 1.0
    else:
        lr_adj = 0.5 * (1 + math.cos(math.pi * epoch / (total_epochs - warm_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj

