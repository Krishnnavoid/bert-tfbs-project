# ========================================================================
# [RUN GUIDE]
# test_cross_cell.py  -  cross-cell-line evaluation (Novelty 2)
#
# Evaluates trained weights on a target test set.
# Automatically detects the exact input_channel from the .pth checkpoint
# and pads the test sequences to match it precisely.
#
# Example run:
#   python -u test_cross_cell.py \
#       --test-dataset  Dataset/wgEncodeAwgTfbsHaibA549Ctcfsc \
#       --weights-path  Dataset/wgEncodeAwgTfbsBroadK562CtcfUniPk/model_full_seed41_Broad_K562.pth \
#       --train-name    K562 \
#       --arch full \
#       --out results/cross_K562_to_A549.csv
# ========================================================================

import argparse
import importlib
import os

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from script.dataloader import MyDataset
import torch.utils.data as Data
from script.metrics import MetricTracker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ARCH = {'full': 'script.model', 'v1': 'script.model_V1', 'v2': 'script.model_V2',
        'resnet': 'script.model_resnet'}

def detect_input_channel(state_dict):
    """Infer the input_channel dimension the weights were trained with."""
    for k, v in state_dict.items():
        if any(layer in k for layer in ['model.conv1.0.weight', 'model.conv1.weight', 'model.stem.0.weight']):
            return v.shape[1]
        elif 'model.linear1.weight' in k and len(v.shape) == 2:
            return v.shape[0]
    return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-dataset', required=True, help='folder containing test.csv of target cell line')
    ap.add_argument('--weights-path', required=True, help='full path to the .pth weights file')
    ap.add_argument('--train-dataset', default=None, help='optional path or name of train dataset')
    ap.add_argument('--train-name', default=None, help='optional label for the training cell line')
    ap.add_argument('--arch', default='full', choices=list(ARCH))
    ap.add_argument('--out', default='results/cross_cell_test.csv', help='output CSV')
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()

    test_name = os.path.basename(os.path.normpath(args.test_dataset))
    train_name = args.train_name or (os.path.basename(os.path.normpath(args.train_dataset)) if args.train_dataset else 'Trained_Weights')

    print('=' * 60)
    print(f'Cross-cell-line evaluation')
    print(f'  Trained on   : {train_name}')
    print(f'  Testing on   : {test_name}')
    print(f'  Weights file : {args.weights_path}')
    print('=' * 60)

    # 1. Load checkpoint first to detect trained channel dimension
    print('Loading weights checkpoint...')
    state_dict = torch.load(args.weights_path, map_location=device)
    input_channel = detect_input_channel(state_dict)
    if input_channel is None:
        raise ValueError('Could not automatically determine input_channel from checkpoint weights.')
    print(f'Checkpoint requires input_channel (token length): {input_channel}')

    # 2. Tokenize target test set
    test_file = os.path.join(args.test_dataset, 'test.csv')
    print(f'Reading {test_file}...')
    test_data = pd.read_csv(test_file)

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    print(f'Tokenizing {len(test_data)} test sequences...')
    test_tokens = [tokenizer(seq, return_tensors='pt')["input_ids"].squeeze() for seq in test_data["sequence"]]
    test_inputs = torch.nn.utils.rnn.pad_sequence(test_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

    # 3. Align padded sequence length to checkpoint's expected input_channel
    curr_len = test_inputs.shape[1]
    if curr_len < input_channel:
        pad_len = input_channel - curr_len
        pad_tensor = torch.full((test_inputs.shape[0], pad_len), tokenizer.pad_token_id, dtype=test_inputs.dtype)
        test_inputs = torch.cat([test_inputs, pad_tensor], dim=1)
        print(f'Padded test tokens from length {curr_len} -> {input_channel}')
    elif curr_len > input_channel:
        test_inputs = test_inputs[:, :input_channel]
        print(f'Truncated test tokens from length {curr_len} -> {input_channel}')
    else:
        print(f'Test token length matches expected input_channel: {input_channel}')

    test_labels = torch.tensor(test_data["label"].values, dtype=torch.long)
    testloader = Data.DataLoader(MyDataset(test_inputs, test_labels), batch_size=128, shuffle=False)

    # 4. Instantiate model and load state_dict
    Bert_Blend_CNN = importlib.import_module(ARCH[args.arch]).Bert_Blend_CNN
    model = Bert_Blend_CNN(input_channel)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Evaluate
    print('Evaluating model on target cell line test set...')
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in testloader:
            x, y = (p.to(device) for p in batch)
            pred = model(x)
            tracker.update(pred, y, loss_fn(pred, y).item())
    m = tracker.compute()

    print('\nEvaluation Results:')
    print('  Acc = %.4f, Prec = %.4f, Rec = %.4f, Spec = %.4f, F1 = %.4f, MCC = %.4f, '
          'ROC-AUC = %.4f, PR-AUC = %.4f'
          % (m['acc'], m['precision'], m['recall'], m['specificity'], m['f1'],
             m['mcc'], m['roc_auc'], m['pr_auc']))
    print('  Confusion matrix  TN=%d FP=%d FN=%d TP=%d' % (m['tn'], m['fp'], m['fn'], m['tp']))

    row = {'train_cell_line': train_name, 'test_cell_line': test_name,
           'arch': args.arch, 'weights': os.path.basename(args.weights_path), **m}
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Saved results to -> {args.out}')
