# ========================================================================
# [RUN GUIDE]
# train_metrics.py  -  train ONE model variant, log metrics every epoch
#
# Run in a WSL terminal, from ~/BERT-TFBS (project is /workspace in the container).
#
#   0. start the container (once per boot):
#        docker compose up -d
#   1. train in the background, output to a log file:
#        docker compose exec -d bert-tfbs sh -c "python -u train_metrics.py --arch full --tag full --seed 41 > /workspace/results_full_41.log 2>&1"
#      --arch   full | v1 | v2 | resnet   which model to train
#      --tag    label for output file names (keep it equal to --arch)
#      --seed   random seed (use 41, 42, 43)
#      --epochs default 15
#   2. watch progress (Ctrl+C stops watching, NOT training):
#        tail -f results_full_41.log
#      is it still running? (no output = finished):
#        docker compose exec bert-tfbs pgrep -af train_metrics
#   3. outputs:
#        Dataset/<folder>/model_<tag>_seed<N>.pth    weights (overwritten every epoch)
#        results/metrics_<tag>_seed<N>.csv           per-epoch train/test metrics
#   4. then test it:  python test.py --arch <arch> --weights model_<tag>_seed<N>.pth
#
# Train ONE job at a time on a 6 GB GPU.
# ========================================================================

import argparse
import os
import random

import numpy as np
import pandas as pd
from transformers import AdamW

from script.adjust_learning import *
from script.dataloader import *
import importlib
import torch
import torch.nn as nn
from script.metrics import MetricTracker, append_csv
# BERT-TFBS-V1 / V2 / ResNet variants: swap the import above, e.g.
# from script.model_V1 import *
# from script.model_resnet import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--tag', default='full', help='name of this model variant (used in output files)')
    ap.add_argument('--arch', default='full', choices=['full', 'v1', 'v2', 'resnet'])
    ap.add_argument('--dataset', default=None, help='run only this specific dataset subfolder')
    args = ap.parse_args()
    ARCH = {'full': 'script.model', 'v1': 'script.model_V1', 'v2': 'script.model_V2', 'resnet': 'script.model_resnet'}
    Bert_Blend_CNN = importlib.import_module(ARCH[args.arch]).Bert_Blend_CNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoches = args.epochs
    dataset_folder = os.environ.get('DATASET_FOLDER', 'Dataset')
    os.makedirs('results', exist_ok=True)
    log_path = os.path.join('results', f'metrics_{args.tag}_seed{args.seed}.csv')

    folders = [args.dataset] if args.dataset else sorted(os.listdir(dataset_folder))
    for folder_name in folders:
        folder_path = os.path.join(dataset_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        set_seed(args.seed)  # re-seed for every dataset so runs are independent
        train_data = pd.read_csv(os.path.join(folder_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(folder_path, 'test.csv'))
        train_inputs, train_labels, test_inputs, test_labels = input_token(
            train_data["sequence"], train_data["label"], test_data["sequence"], test_data["label"])
        input_channel = len(train_inputs[1])
        trainloader = Data.DataLoader(MyDataset(train_inputs, train_labels), batch_size=32, shuffle=True)
        testloader = Data.DataLoader(MyDataset(test_inputs, test_labels), batch_size=32, shuffle=False)

        model = Bert_Blend_CNN(input_channel).to(device)
        optimizer = AdamW(model.parameters(), lr=1.5e-5, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        print('%s: tag=%s seed=%d' % (folder_name, args.tag, args.seed))

        for epoch in range(epoches):
            # ---- training pass ----
            model.train()
            tr = MetricTracker()
            for batch in trainloader:
                x, y = (p.to(device) for p in batch)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epoches,
                                     lr_min=2e-6, lr_max=1.5e-5, warmup=True)
                optimizer.step()
                tr.update(pred, y, loss.item())
            tr_m = tr.compute()

            # ---- evaluation pass (eval mode, no gradients) ----
            model.eval()
            te = MetricTracker()
            with torch.no_grad():
                for batch in testloader:
                    x, y = (p.to(device) for p in batch)
                    pred = model(x)
                    te.update(pred, y, loss_fn(pred, y).item())
            te_m = te.compute()

            print('epoch %2d | lr %.2e | train loss %.4f acc %.4f | test loss %.4f acc %.4f '
                  'F1 %.4f MCC %.4f ROC %.4f PR %.4f'
                  % (epoch + 1, optimizer.param_groups[0]['lr'], tr_m['loss'], tr_m['acc'],
                     te_m['loss'], te_m['acc'], te_m['f1'], te_m['mcc'], te_m['roc_auc'], te_m['pr_auc']),
                  flush=True)
            row = {'dataset': folder_name, 'tag': args.tag, 'seed': args.seed, 'epoch': epoch + 1,
                   'lr': optimizer.param_groups[0]['lr']}
            row.update({'train_' + k: v for k, v in tr_m.items()})
            row.update({'test_' + k: v for k, v in te_m.items()})
            append_csv(log_path, row)

            # keep the latest weights (overwritten each epoch)
            torch.save(model.state_dict(), os.path.join(folder_path, f'model_{args.tag}_seed{args.seed}.pth'))
