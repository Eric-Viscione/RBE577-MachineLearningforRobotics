from sklearn.datasets import fetch_california_housing ##import the dataset that this will be trained and evaluated against




import torch.nn as nn #import the neural network and the pytorch framework and have it be referenced as nn in the code
import torch.optim as optim #import the pytoch optimizer and have it called as optim
import torch
import copy #allows for deep or shallow copies of objects
import numpy as np #imports the numpy library and references it as numpy
import torch
import matplotlib.pyplot as plt  #imports the matplot lib library for plotting the results and referencing it as plt
from sklearn.model_selection import train_test_split #importsd a function to split the larger data set into a training and a test data set

lr = 0.0001
train_size = 0.7
layer_size =8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #a check to tell the neural network which processing device to use either gpu or cpu
data = fetch_california_housing() ##pull the data from the dataset and sklearn library
print(data.feature_names) #prints the headers from the data 
X, y = data.data, data.target ##this assigns the features of the data set to the X variable(things that cvna be used to predict the finalr result, the independent variables) and y is the final answer we are looking for(the dependent variable)
# print(y)
# print(X)
#this defines a sequential set of layers so the first layer is an 8 input with an output of 24 and each subsequent layer must match the next one in line, the output is one for the final layer because we are just looking for one value, price. If we wanted to find price and sqft we could change the X,y data to represent that and then change the final layer to two
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
).to(device)  # Move model to device
loss_fn = nn.MSELoss() #define the loss function we are going to use as the mean square error function
optimizer = optim.SGD(model.parameters(), lr=0.0001) #stochastic gradient descent optimizer, model parameters gives the optimiszer the info about the network abocve, the lr is the learning rate which is how fast the optimizer will tweak the weights
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, shuffle=True) #this is from the sklearn library and is going to split the X and y data into two groups, 70% will be used for the training and 30% for testing and then shuffling it randomizes the data order
X_train = torch.tensor(X_train, dtype=torch.float32).to(device) #move the data used for training to the device
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1).to(device) #you reshape this data to match the size of our output in this case 1, and the -1 basically tell the function to base the size of the result on the other value. so inn this case it will be an Zx1 vector
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1,1).to(device)

#training parameters
n_epochs = 100 #the number of epochs or cycles this shoudl train for
batch_size = 100 #the size of each batch
batch_start = torch.arange(0,len(X_train), batch_size) #this creates a tensor with the point each batch will start starts at 0 to the length of the x_Train data and it will be spaced based on the batch size variable in this case every 100 so [0, 100, 200, 300 .......... len(X_test)]

#Hold the best model 
best_mse = np.inf #initialize the best result to infinity at first
best_weights = None #we dont have any weights yet
history_eval = []
history_train = []

for epoch in range(n_epochs): #for loop that will go once through each epoch
    model.train() #this is important and sets the model to learning mode could be set to eval to test the model when I am done
    print(f"Epoch {epoch+1}/{n_epochs}")
    for start in batch_start: #loops through that set of starting values we had earlier
        X_batch = X_train[start:start+batch_size] #this takes the value of the the indicy we are at and adds the batch size and pulls those values from our training data (for the first iterations start = 0 so [0:100])
        y_batch = y_train[start:start+batch_size]
        #forward_pass
        y_pred = model(X_batch) #this passes the data through the current model and then stores the predicted output
        loss = loss_fn(y_pred, y_batch)#.item()turn this on after I confirm this works #this represents the loss of this pass so basically the accuracy
        history_train.append(loss.item()) #this converts the tensor value to a float that we can evaluate later
        #backward pass
        optimizer.zero_grad() #clears the gradients from the previous pass
        loss.backward() #computes the new gradients
        optimizer.step()#update the weights of the model
        print(f" epoch: {epoch+1}/{n_epochs}, Batch {start//batch_size+1}/{len(batch_start)}, Loss:{loss.item():.6f}")
    model.eval #sets the model to evaluation mode so we can test how that pass did
    with torch.no_grad(): #we turn off the gradient computations because we are testign the model and not optimizing it
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test).item()
        history_eval.append(mse)
        if mse <best_mse: #this is checking to see if we have a better model than the previous best
            best_mse = mse #stores that
            best_weights = copy.deepcopy(model.state_dict())#this copies the weights and stores them

model.load_state_dict(best_weights) #sets out model to use the best weights we found during training\
print(f"Best MSE: {best_mse:.2f}")
print(f"Best RMSE: {np.sqrt(best_mse):.2f}")
plt.figure(1)
plt.plot(history_eval)
plt.xlabel('Epoch')
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
plt.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='horizontal Line')
plt.figure(2)
plt.plot(history_train)
plt.title("MSE in train")

plt.show
# type(history_eval)# for epoch in range(history_eval):
print(len(history_eval))
for x in range(len(history_eval)):
    print(f"{x}: {history_eval[x]}")
for x in range(len(history_eval)):
    if history_eval[x]< 1.3:
        print(f"The value where x crosses the min threshhold is {x}")
        break
        

