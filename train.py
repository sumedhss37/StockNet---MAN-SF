from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle

from utils import load_data, accuracy  # load_data: load relation data
from models import GAT

def print_param_sums(model, loss_value, threshold):
    # Calculate the total sum of parameters that require gradients.
    total_sum = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_sum += param.sum().item()

    print(f"\n=== Loss crossed threshold {threshold}: current loss = {loss_value:.4f} ===")
    print(f"Total sum of parameters: {total_sum:.4f}")

def check_gradients(model):
    # Check and print if any parameter has zero or None gradient.
    for name, param in model.named_parameters():
        if param.grad is None:
            # print(f"{name} has no gradient!")
            pass
        elif torch.all(param.grad == 0):
            # print(f"{name} has zero gradient!")
            pass

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
# Lower weight decay to prevent over-regularization (adjust as needed)
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.38, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()

# Use MPS if available, otherwise use CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif not args.no_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set random seeds for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed(args.seed)
# Note: For MPS, no special manual seed call is required.

# Load data for GAT
adj = load_data()
stock_num = adj.size(0)

train_price_path = "./Data/train_price/"
train_label_path = "./Data/train_label/"
train_text_path = "./Data/train_text/"
test_price_path = "./Data/test_price/"
test_label_path = "./Data/test_label/"
test_text_path = "./Data/test_text/"
num_samples = len(os.listdir(train_price_path))

import matplotlib.pyplot as plt

# Define cross-entropy loss (ensure the weight tensor is on the same device)
cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.00, 1.00]).to(device))

# Define the thresholds (sorted in descending order if loss decreases)
thresholds = [2000, 1000, 800, 400, 300, 200, 100, 50, 25, 15, 5, 1]
triggered_thresholds = set()

def train(epoch, TRAIN_SIZE):
    global triggered_thresholds  # so we can update the set across epochs
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # Randomly sample one training sample
    i = np.random.randint(TRAIN_SIZE)
    train_text = np.load(train_text_path + str(i).zfill(10) + '.npy')
    train_price = np.load(train_price_path + str(i).zfill(10) + '.npy')
    train_label = np.load(train_label_path + str(i).zfill(10) + '.npy')
    
    # Log the shapes for debugging
    # print(f"Train sample {i}: text shape {train_text.shape}, price shape {train_price.shape}, label shape {train_label.shape}")

    # Convert to tensors and send to device
    train_text = torch.tensor(train_text, dtype=torch.float32).to(device)
    train_price = torch.tensor(train_price, dtype=torch.float32).to(device)
    train_label = torch.LongTensor(train_label).to(device)

    output = model(train_text, train_price, adj)
    loss_train = cross_entropy(output, train_label)
    acc_train = accuracy(output, train_label)
    loss_train.backward()

    # Check gradients after backward pass
    # check_gradients(model)

    optimizer.step()

    # Check if loss_train has fallen below any threshold that hasn't been printed yet
    current_loss = loss_train.item()
    for threshold in thresholds:
        # Adjust condition if needed; here we check if loss is greater than a threshold.
        if current_loss > threshold and threshold not in triggered_thresholds:
            print_param_sums(model, current_loss, threshold)
            triggered_thresholds.add(threshold)
    
    if epoch % 10 == 0:
        print("Epoch:", epoch, ", Training loss =", current_loss, ", Accuracy =", acc_train.item(), ", Time taken =", time.time()-t)

def test(TEST_SIZE):
    model.eval()
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    with torch.no_grad():
        for i in range(TEST_SIZE):
            test_text = np.load(test_text_path + str(i).zfill(10) + '.npy')
            test_price = np.load(test_price_path + str(i).zfill(10) + '.npy')
            test_label = np.load(test_label_path + str(i).zfill(10) + '.npy')
            
            # Log shapes for debugging (you may comment out after verification)
            # print(f"Test sample {i}: text shape {test_text.shape}, price shape {test_price.shape}, label shape {test_label.shape}")

            test_text = torch.tensor(test_text, dtype=torch.float32).to(device)
            test_price = torch.tensor(test_price, dtype=torch.float32).to(device)
            test_label = torch.LongTensor(test_label).to(device)
            
            output = model(test_text, test_price, adj)
            loss_test = cross_entropy(output, test_label)
            acc_test = accuracy(output, test_label)
            a = output.argmax(1).cpu().numpy()
            b = test_label.cpu().numpy() 
            li_pred.append(a)
            li_true.append(b)
            test_loss.append(loss_test.item())
            test_acc.append(acc_test.item())
    f1 = f1_score(np.array(li_true).reshape(-1), np.array(li_pred).reshape(-1), average='micro')
    mcc = matthews_corrcoef(np.array(li_true).reshape(-1), np.array(li_pred).reshape(-1))
    print("Test set results:",
          "loss= {:.4f}".format(np.array(test_loss).mean()),
          "accuracy= {:.4f}".format(np.array(test_acc).mean()),
          "F1 score= {:.4f}".format(f1),
          "MCC = {:.4f}".format(mcc))

# Instantiate model. Ensure that the modifications in your models.py (e.g., using nn.ModuleList)
# are in place so that all parameters are registered.
model = GAT(nfeat=64, 
            nhid=args.hidden, 
            nclass=2, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha,
            stock_num=stock_num)
model.to(device)
adj = adj.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

TRAIN_SIZE = 300
TEST_SIZE = 90
for epoch in range(args.epochs):
    train(epoch, TRAIN_SIZE)
    if epoch % 100 == 0:
        torch.save(model.state_dict(), "./weight.pth")
        test(TEST_SIZE)
print("Optimization Finished!")
