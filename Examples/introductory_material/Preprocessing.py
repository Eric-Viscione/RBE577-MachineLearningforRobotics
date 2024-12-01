import os
import pandas as pd
import torch
from ucimlrepo import fetch_ucirepo 
import psutil


def possible_data(data_set):
    mem = data_set.data.features.memory_usage(index=True, deep=False)
    integer_gig = int(mem.sum())/(1024**3)
    print(f"The size of this data set is {integer_gig}")
    memory_info = psutil.virtual_memory().total
    print(f"Total System Memory: {memory_info / (1024**3):.2f} GB")
    num_sets_possible = memory_info/integer_gig
    print(f"The toal number of datasets my computer can handle is {num_sets_possible}")




os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
    
data = pd.read_csv(data_file)
# print("The raw data")
# print(data)
# print("/////////")
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
inputs = inputs.fillna(inputs.mean())
# print(inputs)
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
# print(X)
# print(y)

##Exercises
Abalone = fetch_ucirepo(id=1) 


#Abalone = Abalone[::20]

# print(Abalone.data.features.head(20))
# print(Abalone.data.targets.head(20))
# print(Abalone.data.targets.dtypes)
# print(Abalone.data.features.head(20).dtypes)
# print(Abalone.data.features.head(20).loc[:, 'Sex'])
possible_data(Abalone)
mem = Abalone.data.features.memory_usage(index=True, deep=False)
# print(mem.loc['Sex'])
integer_gig = int(mem.sum())/(1024**3)
print(integer_gig)
memory_info = psutil.virtual_memory().total
print(f"Total Memory: {memory_info / (1024**3):.2f} GB")
num_sets_possible = memory_info/integer_gig
print(f"The toal number of datasets my computer can handle is {num_sets_possible}")
data_sets = [300, 352, 52, 602, 45, 2, 109]
for i in range(len(data_sets)):
    
    print("/////////////////")
    print(f"Data Set: {data_sets[i]}")
    dataset = fetch_ucirepo(id=data_sets[i]) 
    possible_data(dataset)
    print("/////////////////")




