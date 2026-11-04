# train_baseline.py - train ONE one-hot benchmark model (DanQ, ...) and log metrics every epoch.
# Same data folders, seeds, metric code and CSV format as train_metrics.py.
#   train :  python -u train_baseline.py --arch danq --tag danq --seed 41 --epochs 10
#   test  :  python -u train_baseline.py --arch danq --tag danq --seed 41 --eval_only
# Outputs
#   results/metrics_<tag>_seed<N>.csv   one row per epoch (train_* and test_* columns, as for BERT-TFBS)
#   results/test_<tag>_seed<N>.csv      one row: final test metrics of the saved weights
#   Dataset/<folder>/model_<tag>_seed<N>.pth   weights (overwritten every epoch)
import argparse
import importlib
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from script.metrics import MetricTracker, append_csv

ARCH = {'danq': ('script.model_danq', 'DanQ')}   # add 'deepbind': (...) later

_LUT = np.full(256, 4, dtype=np.int64)            # A,C,G,T -> 0..3 ; anything else (N) -> 4
for _i, _c in enumerate('ACGT'):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


def one_hot(seqs, length):
    out = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for i, s in enumerate(seqs):
        idx = _LUT[np.frombuffer(s.encode('ascii'), dtype=np.uint8)][:length]
        pos = np.nonzero(idx < 4)[0]
        out[i, idx[pos], pos] = 1.0               # N / short sequences stay all-zero
    return torch.from_numpy(out)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    te = MetricTracker()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            te.update(pred, y, loss_fn(pred, y).item())
    return te.compute()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--arch', default='danq', choices=list(ARCH))
    ap.add_argument('--tag', default=None, help='label for output files (default: same as --arch)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_min', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--data', default='Dataset')
    ap.add_argument('--eval_only', action='store_true', help='skip training; evaluate the saved weights on test.csv')
    ap.add_argument('--max_train', type=int, default=0, help='debug: use only the first N training rows')
    args = ap.parse_args()
    tag = args.tag or args.arch

    module, cls = ARCH[args.arch]
    Model = getattr(importlib.import_module(module), cls)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)
    log_path = os.path.join('results', f'metrics_{tag}_seed{args.seed}.csv')
    test_path = os.path.join('results', f'test_{tag}_seed{args.seed}.csv')
    if os.path.exists(test_path):
        os.remove(test_path)          # one fresh row per dataset on every run

    for folder_name in sorted(os.listdir(args.data)):
        folder = os.path.join(args.data, folder_name)
        if not os.path.isdir(folder):
            continue
        weights = os.path.join(folder, f'model_{tag}_seed{args.seed}.pth')
        set_seed(args.seed)
        train = pd.read_csv(os.path.join(folder, 'train.csv'))
        test = pd.read_csv(os.path.join(folder, 'test.csv'))
        if args.max_train:
            train = train.iloc[:args.max_train]
        seq_len = int(train['sequence'].str.len().max())
        xtr, xte = one_hot(train['sequence'], seq_len), one_hot(test['sequence'], seq_len)
        ytr = torch.tensor(train['label'].values, dtype=torch.long)
        yte = torch.tensor(test['label'].values, dtype=torch.long)
        trainloader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch, shuffle=True)
        testloader = DataLoader(TensorDataset(xte, yte), batch_size=args.batch, shuffle=False)

        model = Model(seq_len=seq_len).to(device)
        loss_fn = nn.CrossEntropyLoss()

        if not args.eval_only:
            n_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            print('%s: arch=%s tag=%s seed=%d epochs=%d params=%.2fM train=%d test=%d len=%d'
                  % (folder_name, args.arch, tag, args.seed, args.epochs, n_params / 1e6,
                     len(ytr), len(yte), seq_len), flush=True)
            for epoch in range(args.epochs):
                t0 = time.time()
                lr = args.lr_min + (args.lr - args.lr_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
                for g in optimizer.param_groups:
                    g['lr'] = lr
                model.train()
                tr = MetricTracker()
                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()
                    tr.update(pred, y, loss.item())
                tr_m = tr.compute()
                te_m = evaluate(model, testloader, loss_fn, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                print('epoch %2d | lr %.2e | train loss %.4f acc %.4f | test loss %.4f acc %.4f '
                      'F1 %.4f MCC %.4f ROC %.4f PR %.4f | %.0fs'
                      % (epoch + 1, lr, tr_m['loss'], tr_m['acc'], te_m['loss'], te_m['acc'],
                         te_m['f1'], te_m['mcc'], te_m['roc_auc'], te_m['pr_auc'], time.time() - t0), flush=True)
                row = {'dataset': folder_name, 'tag': tag, 'seed': args.seed, 'epoch': epoch + 1, 'lr': lr}
                row.update({'train_' + k: v for k, v in tr_m.items()})
                row.update({'test_' + k: v for k, v in te_m.items()})
                append_csv(log_path, row)
                torch.save(model.state_dict(), weights)

        # final test evaluation of the saved (= last-epoch) weights
        model.load_state_dict(torch.load(weights, map_location=device))
        te_m = evaluate(model, testloader, loss_fn, device)
        row = {'dataset': folder_name, 'tag': tag, 'seed': args.seed}
        row.update({'test_' + k: v for k, v in te_m.items()})
        append_csv(test_path, row)
        print('FINAL TEST %s | acc %.4f F1 %.4f MCC %.4f ROC %.4f PR %.4f  -> %s'
              % (folder_name, te_m['acc'], te_m['f1'], te_m['mcc'], te_m['roc_auc'], te_m['pr_auc'], test_path), flush=True)
