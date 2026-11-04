# ========================================================================
# [RUN GUIDE]
# test.py  -  evaluate trained weights on the test set (accuracy, precision, recall,
#             specificity, F1, MCC, ROC-AUC, PR-AUC, confusion matrix)
#
# Run in a WSL terminal, from ~/BERT-TFBS.  Needs the container up:  docker compose up -d
#
#   Test a model trained with train_metrics.py (--arch must match the training run):
#      docker compose exec bert-tfbs python test.py --arch full   --weights model_full_seed41.pth   --out results/test_full_seed41.csv
#      docker compose exec bert-tfbs python test.py --arch v1     --weights model_v1_seed41.pth     --out results/test_v1_seed41.csv
#      docker compose exec bert-tfbs python test.py --arch v2     --weights model_v2_seed41.pth     --out results/test_v2_seed41.csv
#      docker compose exec bert-tfbs python test.py --arch resnet --weights model_resnet_seed41.pth --out results/test_resnet_seed41.csv
#
#   Test the original author's checkpoint (Dataset/<folder>/model.pth):
#      docker compose exec bert-tfbs python test.py
#
#   --arch     full | v1 | v2 | resnet
#   --weights  file name inside Dataset/<folder>/   (default model.pth)
#   --out      results table (CSV)
#
# Weights not found / a Git LFS pointer is skipped with a message.
# Do not run tests while a training job is using the GPU.
# ========================================================================

import argparse
import importlib
import os

import pandas as pd
import torch
import torch.nn as nn

from script.dataloader import *
from script.metrics import MetricTracker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# model variant -> module that defines Bert_Blend_CNN
ARCH = {'full': 'script.model', 'v1': 'script.model_V1', 'v2': 'script.model_V2',
        'resnet': 'script.model_resnet'}
RATE_COLS = ['loss', 'acc', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'roc_auc', 'pr_auc']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='full', choices=list(ARCH), help='model variant')
    ap.add_argument('--weights', default='model.pth', help='weights file name inside each Dataset/<folder>/')
    ap.add_argument('--dataset', default=None, help='run only this specific dataset subfolder')
    ap.add_argument('--out', default='results/test_metrics.csv', help='where to save the results table')
    args = ap.parse_args()

    Bert_Blend_CNN = importlib.import_module(ARCH[args.arch]).Bert_Blend_CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_folder = os.environ.get('DATASET_FOLDER', 'Dataset')
    loss_fn = nn.CrossEntropyLoss()
    rows = []

    folders = [args.dataset] if args.dataset else sorted(f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f)))
    with torch.no_grad():
        for folder_name in folders:
            folder_path = os.path.join(dataset_folder, folder_name)
            model_file = os.path.join(folder_path, args.weights)
            if not os.path.isfile(model_file):
                print('%s: %s not found, skipping' % (folder_name, args.weights))
                continue
            if os.path.getsize(model_file) < 1_000_000:
                print('%s: %s is only %d bytes (a Git LFS pointer?), skipping'
                      % (folder_name, args.weights, os.path.getsize(model_file)))
                continue
            train_data = pd.read_csv(os.path.join(folder_path, 'train.csv'))
            test_data = pd.read_csv(os.path.join(folder_path, 'test.csv'))
            # tokenise and pad (train+test are padded together, as in training)
            train_inputs, train_labels, test_inputs, test_labels = input_token(
                train_data["sequence"], train_data["label"], test_data["sequence"], test_data["label"])
            input_channel = len(train_inputs[1])
            testloader = Data.DataLoader(MyDataset(test_inputs, test_labels), batch_size=128, shuffle=False)

            model = Bert_Blend_CNN(input_channel)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()

            tracker = MetricTracker()
            for batch in testloader:
                x, y = (p.to(device) for p in batch)
                pred = model(x)
                tracker.update(pred, y, loss_fn(pred, y).item())
            m = tracker.compute()
            rows.append({'dataset': folder_name, 'arch': args.arch, 'weights': args.weights, **m})
            print('%s: Acc = %.4f, Prec = %.4f, Rec = %.4f, Spec = %.4f, F1 = %.4f, MCC = %.4f, '
                  'ROC-AUC = %.4f, PR-AUC = %.4f'
                  % (folder_name, m['acc'], m['precision'], m['recall'], m['specificity'], m['f1'],
                     m['mcc'], m['roc_auc'], m['pr_auc']))
            print('  confusion matrix  TN=%d FP=%d FN=%d TP=%d' % (m['tn'], m['fp'], m['fn'], m['tp']))

    if not rows:
        raise SystemExit('No datasets were evaluated. Check the weights file name and location.')

    df = pd.DataFrame(rows)
    avg = {'dataset': 'average (%d datasets)' % len(df), 'arch': args.arch, 'weights': args.weights}
    avg.update({c: df[c].mean() for c in RATE_COLS})
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    print('mean over %d dataset(s): Acc = %.4f, ROC-AUC = %.4f, PR-AUC = %.4f'
          % (len(rows), avg['acc'], avg['roc_auc'], avg['pr_auc']))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print('saved ->', args.out)
