import math as m 
import numpy as np
import random as rand
import pandas as pd



def generate_data():
   
    F1 = [-10000, 10000]
    F2 = [-5000,  5000]
    F3 = [-5000,  5000]
    alpha2 = [-180, 180]
    alpha3 = [-180, 180]
    ranges = [F1, F2, F3, alpha2, alpha3]
    l1 = -14
    l2 = 14.5
    l3 = -2.7
    l4 = 2.7
    F1_rand = rand.randint(F1[0], F1[1])
    F2_rand = rand.randint(F2[0], F2[1])
    F3_rand = rand.randint(F3[0], F3[1])
    alpha2_rand = rand.randint(alpha2[0], alpha2[1])
    alpha3_rand = rand.randint(alpha3[0], alpha3[1])


    alpha2_cos = m.cos(m.radians(alpha2_rand))
    alpha2_sin = m.sin(m.radians(alpha2_rand))
    alpha3_cos = m.cos(m.radians(alpha3_rand))
    alpha3_sin = m.sin(m.radians(alpha3_rand))

    u = np.array([[F1_rand], [F2_rand], [alpha2_rand], [F3_rand], [alpha3_rand]])
    # print(u)
    # print(u.shape)
    forces = np.array([[F1_rand], [F2_rand], [F3_rand]])
    geometry = np.array([[0, alpha2_cos, alpha3_cos], 
                    [1, alpha2_sin, alpha3_sin], 
                    [l2, ((l1*alpha2_sin)-(l3*alpha2_cos)), ((l1*alpha3_sin)-(l4*alpha3_cos))]])
    tau= np.dot(geometry, forces)
    # print(tau)
    # print(tau.shape)
    tau_with_inputs = np.concatenate((tau,u ), axis=0)
    return tau_with_inputs.flatten()


num_datapoints = 5000
df = pd.DataFrame(columns=['Tau_Surge', 'Tau_Sway', 'Tau_Yaw', 'F1', 'F2', 'Alpha2', 'F3', 'Alpha3'])   
for i in range(num_datapoints):
    print(f"{(i/num_datapoints)*100}%")
    data = generate_data()
    new_row = pd.DataFrame([data], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)

print(df)
df.to_pickle("~/Desktop/RBE577-MachineLearningforRobotics/homework1/Training_Data.pkl")

df_check = pd.read_pickle("~/Desktop/RBE577-MachineLearningforRobotics/homework1/Training_Data.pkl")
print(df_check)
if df.equals(df_check):
    print("The DataFrames are equal.")
else:
    print("The DataFrames are not equal.")    
    
    
























