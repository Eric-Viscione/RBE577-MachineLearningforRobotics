import torch.nn as nn #import the neural network and the pytorch framework and have it be referenced as nn in the code
import torch.optim as optim #import the pytoch optimizer and have it called as optim
import torch
import copy #allows for deep or shallow copies of objects
import numpy as np #imports the numpy library and references it as numpy
import torch
import matplotlib.pyplot as plt  #imports the matplot lib library for plotting the results and referencing it as plt
import math as m 
from torch.utils.data import random_split, DataLoader, TensorDataset
import pandas as pd
import time


def write_readme():

        print(f"Pytorch version: {torch.__version__}\n")  # Write PyTorch version
        print(f"NumPy version: {np.__version__}\n")  # Write NumPy version
        print(f"Pandas version: {pd.__version__}\n")  # Write Pandas version


def compute_ycmd(u):

    """
    Computes the Torques based off the inputs from the thrusters

    Inputs: u, the tensor of all the force commands from a given batch, F1 F2 A2 F3 A3

    Outputs: batch size x 3 tensor of the torques surge sway and yaw
    """
    # Initialize tensor to store results
    tau_results = torch.zeros((u.size(0), 3), device=u.device)
    
    # Constants
    l1 = -14
    l2 = 14.5
    l3 = -2.7
    l4 = 2.7
    deg_to_rad = torch.tensor(m.pi / 180, device=u.device)

    F1, F2, alpha2, F3, alpha3 = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]

    alpha2_rad = alpha2 * deg_to_rad
    alpha3_rad = alpha3 * deg_to_rad

   
    alpha2_cos = torch.cos(alpha2_rad)
    alpha2_sin = torch.sin(alpha2_rad)
    alpha3_cos = torch.cos(alpha3_rad)
    alpha3_sin = torch.sin(alpha3_rad)

    
    forces = torch.stack([F1, F2, F3], dim=1).unsqueeze(2)  # Shape: (batch_size, 3, 1)
    #print(u.shape)
    size = F1.shape[0]
    #print(forces.shape)
    #print(f"Size of batch in loss fn {size}")
    geometry = torch.stack([
        torch.stack([torch.zeros(size, device=u.device), torch.ones(size, device=u.device), torch.full((size,), l2, device=u.device)], dim=1),  # 1st column: [0, 1, l2]
        torch.stack([alpha2_cos, alpha2_sin, (l1 * alpha2_sin - l3 * alpha2_cos)], dim=1),  # 2nd column: [cos(alpha2), sin(alpha2), l1*sin(alpha2) - l3*cos(alpha2)]
        torch.stack([alpha3_cos, alpha3_sin, (l1 * alpha3_sin - l4 * alpha3_cos)], dim=1)   # 3rd column: [cos(alpha3), sin(alpha3), l1*sin(alpha3) - l4*cos(alpha3)]
    ], dim=1)  # Shape: (size, 3, 3)
    
    # Perform batch matrix multiplication to compute tau
    tau = torch.bmm(geometry, forces).squeeze(2)  # Shape: (batch_size, 3)
    tau_results = tau.T  # Transpose to match the original shape
    #print(tau_results.shape)
    return tau_results.transpose(0, 1)
def loss_function(taus_target, u_output, taus_output, limits, rate_limits, ks):
    """
    Calculates the loss based up criteria such as thruster limits, thruster angle limits,
      rate change limits closeness between the decoded values and the given values for the U values and the tau values
    Inputs: taus_target: the expected value of tau, u_output: the u values from the encoder layers, 
            taus_output: values for torque from the decoder layers, limits: the limits of the thrusters and actuators, 
            rate_limits: limts on the rate of change for thrusters , ks: Coefficents of each lostt function
    Outputs: Loss: The loss value calculated from the 5 parameters and weighted via k
    """
    Ls = []  ##initilaize an array to store each one as its calculated
    #print(f"All the shapes for the loss function{taus_target.shape, u_output.shape, taus_output.shape, limits.shape, rate_limits.shape}")
    ycmd = compute_ycmd(u_output)
    #mean square error of the targert tau and the calcuated taus from the u outputs(encoder Layer)
    #print(ycmd.shape)
    Ls0 = torch.mean((taus_target-ycmd)**2)
    Ls.append(Ls0)
    #mean square error of the target tau and the output taus (decoder layer)
    Ls1 = torch.mean((taus_target-taus_output)**2)
    Ls.append(Ls1)
    #check each u value for being to far from the limits
    Ls2 = 0
    Ls3 = 0
    for j in range(len(limits)):
            #s2 += torch.sum(torch.maximum(torch.abs(u_output[i])-limits[i]), 0)  
            Ls2 += torch.sum(torch.maximum(torch.abs(u_output[:,j]) - limits[j], torch.tensor(0.0, device=u_output.device)))
    Ls.append(Ls2)

    u_output_prev = u_output.clone()
    u_output_prev[:,1:] = u_output[:,:-1]
    u_output_prev[:,0] = 0
    ##check each of the u values for it having a very high rate of change
    for k in range(len(rate_limits)):
            Ls3 += torch.sum(torch.maximum(torch.abs(u_output[:,k] - u_output_prev[:,k]) - rate_limits[k], torch.tensor(0.0, device=u_output.device)))   
    Ls.append(Ls3)
    constrained_angles = [[-100, -80], [80, 100]] ##Tuples for the angle constraints

    a2 = u_output[:,2]
    a3 = u_output[:,4]
    mask_a2 = torch.logical_and(a2 > constrained_angles[0][0], a2 < constrained_angles[0][1])
    mask_a3 = torch.logical_and(a3 > constrained_angles[1][0], a3 < constrained_angles[1][1])

    if mask_a2.any():
         Ls4 = Ls1
    else: 
         Ls4 = 0
    if mask_a3.any():
         Ls5 = Ls1    
    else: 
         Ls5 = 0       
    Ls.append(Ls4)
    Ls.append(Ls5)

    L = 0
    for i in range(len(Ls)):   #calculates the final added value for the Ls
            L += ks[i]  * Ls[i]
    return L

##Hyperparmeters for the model
hyperparameters ={
     'num_epochs': 100, 
     'batch_size': 500,
     'learning_rate': .0001,
     'loss_weights': [0.000000001,0.000000001,0.000000001,0.000000001,0.000000001,0.000000001],
     'limits': np.array( [3000 , 3000, 3.1415, 30000 ,3.1415]),
     'rate_limits':np.array([1000, 1000, .1745, 1000, .1745 ]),
     'constrained_angles': [[-100, -80], [80, 100]],
     'optimizer': ['Adem'],
     'regulizer': ['l2_Decay'],
     'weight_decay_regularization': 1e-5


}
start_time = time.time()
#NN settings
num_epochs = hyperparameters['num_epochs']
lr = hyperparameters['learning_rate']
batch_size = hyperparameters['batch_size']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #a check to tell the neural network which processing device to use either gpu or cpu

write_readme()
#Data import and Processing
path_to_data = "~/Desktop/RBE577-MachineLearningforRobotics/homework1/Training_Data_1mil.pkl"
df = pd.read_pickle(path_to_data)  ##read in the previosuly generated data
weight_decay = hyperparameters['weight_decay_regularization']
#print(list(df)) #print the headers of the data to check them
dataset = TensorDataset(torch.tensor(df.values).float())  ##I might need to wrap this in a TensorDataset
print(len(dataset))
train_size = int(0.8 *len(dataset))  ##get my value for 80% of this dataset)
validation_size = len(dataset) - train_size #get the value for validation
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])
train_data_tensor = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
val_data_tensor = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
taus_train = train_data_tensor[:, :3]  #  Tau_Surge, Tau_Sway, Tau_Yaw
u_train = train_data_tensor[:, 3:]     # 1, F2, Alpha2, F3, Alpha3
taus_val = val_data_tensor[:, :3]      
u_val = val_data_tensor[:, 3:] 



##Model Limits and Information
limits = hyperparameters['limits']  ##the limits on F1 F2 A2 F3 A3
ks = hyperparameters['loss_weights']
rate_limits = hyperparameters['rate_limits']  ##Rate limits on F1 F2 A2 F3 A3
constrained_angles = hyperparameters['constrained_angles'] ##Tuples for the angle constraints



##Encoder Decoder Models
Encoder = nn.Sequential(
    nn.Linear(3, 24),   #Layer 1 Input of motion commands
    nn.ReLU(),
    nn.Linear(24, 12),  #Layer 2
    nn.ReLU(),
    nn.Linear(12, 5),   #Layer 3 outputs the 3 Force and 2 Angle Commands
).to(device) 
Decoder = nn.Sequential(
    nn.Linear(5, 24),   #Layer 4 Input of Force and angle vectors 
    nn.ReLU(),
    nn.Linear(24, 12),  #Layer 5
    nn.ReLU(),
    nn.Linear(12, 3),   #Layer 6 outputs the 3 taus
    ).to(device)  # Move model to device



##Load in the optimizer
# optimizer_encoder = optim.SGD(Encoder.parameters(), lr=lr) #stochastic gradient descent optimizer, model parameters gives the optimiszer the info about the network abocve, the lr is the learning rate which is how fast the optimizer will tweak the weights
# optimizer_decoder = optim.SGD(Decoder.parameters(), lr=lr) #stochastic gradient descent optimizer, model parameters gives the optimiszer the info about the network abocve, the lr is the learning rate which is how fast the optimizer will tweak the weights
optimizer_encoder = optim.Adam(Encoder.parameters(), lr=lr,weight_decay=weight_decay)  
optimizer_decoder = optim.Adam(Decoder.parameters(), lr=lr,weight_decay=weight_decay)  # Replace SGD with Adam

batch_start = torch.arange(0,len(taus_train), batch_size) #this creates a tensor with the point each batch will start starts at 0 to the length of the x_Train data and it will be spaced based on the batch size variable in this case every 100 so [0, 100, 200, 300 .......... len(X_test)]
##Initialize values for storing our weights and history
best_mse = np.inf #initialize the best result to infinity at first
best_weights = None #we dont have any weights yet
history_eval = []
history_train = []


for epoch in range(num_epochs): #for loop that will go once through each epoch
    Encoder.train() #this is important and sets the model to learning mode could be set to eval to test the model when I am done
    Decoder.train()
    print(f"Epoch {epoch+1}/{num_epochs}")
    for start in batch_start: #loops through that set of starting values we had earlier
        tau_batch = taus_train[start:start+batch_size].to(device) #this takes the value of the the indicy we are at and adds the batch size and pulls those values from our training data (for the first iterations start = 0 so [0:100])\
        #print(f"Batch size: {tau_batch.size(0)}")

        u_pred = Encoder(tau_batch)  #train the encoder layer to output thruster values
        tau_pred = Decoder(u_pred)   #train the decoder layer to output tau values
        loss = loss_function(tau_batch, u_pred, tau_pred, limits, rate_limits, ks)  
        #print(loss)
        history_train.append(loss.item())  #store the loss value for later

        #clear the gradients from the optimizers
        optimizer_encoder.zero_grad() 
        optimizer_decoder.zero_grad()

        loss.backward()  #computen the new gradients

        #update the weights for the model
        optimizer_encoder.step() 
        optimizer_decoder.step()

        print(f" epoch: {epoch+1}/{num_epochs}, Batch {start//batch_size+1}/{len(batch_start)}, Loss:{loss.item():.6f}")

        #Set the model to Eval Mode
        Encoder.eval()
        Decoder.eval()
        with torch.no_grad(): #Turn off the gradient for testing
              taus_val = taus_val.to(device)
              u_pred_val = Encoder(taus_val)
              tau_pred_eval = Decoder(u_pred_val)
              loss_eval = loss = loss_function(taus_val, u_pred_val, tau_pred_eval, limits, rate_limits, ks)
              history_eval.append(loss.item())  
              if loss < best_mse: #this is checking to see if we have a better model than the previous best
                best_mse = loss_eval #stores that
                best_encoder_weights = copy.deepcopy(Encoder.state_dict())
                best_decoder_weights = copy.deepcopy(Decoder.state_dict())




elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(f"Best MSE: {best_mse.cpu().item():.2f}")
print(f"Best RMSE: {np.sqrt(best_mse.cpu().item()):.2f}")
plt.figure(1)
plt.plot(history_eval)
plt.xlabel('Epoch')
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
#plt.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='Horizontal Line')

plt.figure(2)
plt.plot(history_train)
plt.title("MSE in train")

plt.show()







