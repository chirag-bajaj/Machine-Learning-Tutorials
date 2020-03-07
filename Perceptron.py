import numpy as np
import pandas as pd

data=pd.read_csv("Perceptron_input.csv")
num=data.to_numpy()
print(num)
print(num.shape[0])

inputs=num.shape[1]-1
weights=10*np.random.rand(inputs,1)
bias=np.random.rand(1)

x=num[:,:inputs]
y=num[:,-1]

print(weights)

alpha=0.1

def hardlim(net):
    if net>1:
        out=1
    else:
        out=0
    return out


for i in range(90):
    for j in range(num.shape[0]):
        net=np.dot(x[j],weights)+bias
        #print("net",net)
        out=hardlim(net)
        error=y[j]-out
        #print("error",error)
        for k in range(len(weights)):
            weights[k]=weights[k]-alpha*(error*x[j][k])
            #print("w",weights[k])
        bias=bias-alpha*(error)

print(hardlim(np.dot(x[-3],weights)+bias))
print(hardlim(np.dot(x[-2],weights)+bias))
print(hardlim(np.dot(x[-1],weights)+bias))
print(weights,bias)
