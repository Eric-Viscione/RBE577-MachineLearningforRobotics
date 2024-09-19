import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from torch.utils.data import TensorDataset, DataLoader,random_split
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ycmd_output function
def ycmd_output(u):
    batch_size = u.size(0)
    tau_results = torch.zeros((batch_size, 3), device=u.device)
    l1, l2, l3, l4 = -14, 14.5, -2.7, 2.7
    deg_to_rad = torch.tensor(np.pi / 180, device=u.device)
    
    for i in range(batch_size):
        F1, F2, alpha2, F3, alpha3 = u[i]
        alpha2_rad = alpha2 * deg_to_rad
        alpha3_rad = alpha3 * deg_to_rad

        alpha2_cos = torch.cos(alpha2_rad)
        alpha2_sin = torch.sin(alpha2_rad)
        alpha3_cos = torch.cos(alpha3_rad)
        alpha3_sin = torch.sin(alpha3_rad)

        forces = torch.tensor([[F1], [F2], [F3]], device=u.device)
        geometry = torch.tensor([
            [0, alpha2_cos, alpha3_cos], 
            [1, alpha2_sin, alpha3_sin], 
            [l2, (l1 * alpha2_sin - l3 * alpha2_cos), (l1 * alpha3_sin - l4 * alpha3_cos)]
        ], device=u.device)
        tau = torch.matmul(geometry, forces)
        tau_results[i] = tau.flatten()
    
    return tau_results

# Define the loss function
def loss_function(u_target, taus_Target, u_output, u_output_prev, taus_output, ks, limits, rate_limits, constraints):
    Ls = []
    ycmd = ycmd_output(u_output)
    Ls0 = torch.mean((taus_Target - ycmd) ** 2)
    Ls.append(Ls0)
    
    Ls1 = torch.mean((taus_Target - taus_output) ** 2)
    Ls.append(Ls1)
    
    Ls2 = torch.sum(torch.maximum(torch.abs(u_output) - limits, torch.tensor(0.0, device=u_output.device)))
    Ls.append(Ls2)
    
    Ls3 = torch.sum(torch.maximum(torch.abs(u_output - u_output_prev) - rate_limits, torch.tensor(0.0, device=u_output.device)))
    Ls.append(Ls3)
    
    L = sum(ks[i] * Ls[i] for i in range(len(Ls)))
    return L

# Basic settings
lr = 0.0001
limits = np.array([30000, 30000, np.pi, 300000, np.pi])
ks = [1, 1, 1, 1, 1]
rate_lim = np.array([1000, 1000, 0.1745, 1000, 0.1745])
n_epochs = 100
batch_size = 500

# Load data
df = pd.read_pickle("~/Desktop/RBE577-MachineLearningforRobotics/homework1/Training_Data.pkl")
dataset = torch.tensor(df.values).float()
dataset = TensorDataset(dataset)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_data_tensor = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
val_data_tensor = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
taus_train = train_data_tensor[:, :3]
u_train = train_data_tensor[:, 3:]
taus_val = val_data_tensor[:, :3]
u_val = val_data_tensor[:, 3:]

# Define models
Encoder = nn.Sequential(
    nn.Linear(3, 15),
    nn.ReLU(),
    nn.Linear(15, 30),
    nn.ReLU(),
    nn.Linear(30, 5)
).to(device)

Decoder = nn.Sequential(
    nn.Linear(5, 30),
    nn.ReLU(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Linear(15, 3)
).to(device)

# Define optimizers
optimizer_encoder = optim.SGD(Encoder.parameters(), lr=lr)
optimizer_decoder = optim.SGD(Decoder.parameters(), lr=lr)

# Training loop
batch_start = torch.arange(0, len(taus_train), batch_size)
best_mse = np.inf
best_weights = None
history_eval = []
history_train = []

for epoch in range(n_epochs):
    Encoder.train()
    Decoder.train()
    print(f"Epoch {epoch + 1}/{n_epochs}")
    U_pred_prev = torch.zeros_like(u_train[0]).to(device)
    
    for start in batch_start:
        tau_batch = taus_train[start:start + batch_size].to(device)
        U_pred = Encoder(tau_batch)
        tau_pred = Decoder(U_pred)
        loss = loss_function(None, tau_batch, U_pred, U_pred_prev, tau_pred, ks, limits, rate_lim, [])
        U_pred_prev = U_pred.detach()
        
        history_train.append(loss.item())
        
        optimizer_decoder.zero_grad()
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()
        
        print(f"Epoch: {epoch + 1}/{n_epochs}, Batch {start // batch_size + 1}/{len(batch_start)}, Loss: {loss.item():.6f}")
    
    Encoder.eval()
    Decoder.eval()
    with torch.no_grad():
        taus_val = taus_val.to(device)
        U_pred_val = Encoder(taus_val)
        tau_pred_eval = Decoder(U_pred_val)
        loss = loss_function(None, tau_pred_eval, U_pred_val, U_pred_prev, tau_pred_eval, ks, limits, rate_lim, [])
        
        history_eval.append(loss.item())
        if loss < best_mse:
            best_mse = loss
            best_encoder_weights = copy.deepcopy(Encoder.state_dict())
            best_decoder_weights = copy.deepcopy(Decoder.state_dict())

# Optional: Save models
# torch.save(best_encoder_weights, 'best_encoder.pth')
# torch.save(best_decoder_weights, 'best_decoder.pth')

# Optional: Plot training and evaluation loss
plt.plot(history_train, label='Training Loss')
plt.plot(history_eval, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()