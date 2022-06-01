
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = np.loadtxt(open('mnist_train_small.csv','rb'),delimiter=',')
labels = data[:,0]
data = data[:,1:]
'''
nomralizing the data
'''
dataN = data/np.max(data)
dataT = torch.tensor(dataN).float()
labelsT = torch.tensor(labels).long()

train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelsT, test_size=0.1)

train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

batch_size = 16
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True, drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

def create_neural_network():

    class NeuralNetwork(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(784,64)

            self.hl1 = nn.Linear(64,32)
            self.hl2 = nn.Linear(32,32)

            self.output = nn.Linear(32,10)

        def forward(self,x) -> None:
            x = F.relu(self.input(x))
            x = F.relu(self.hl1(x))
            x = F.relu(self.hl2(x))
            x = self.output(x)
            x = torch.log_softmax(x, axis=1)
            return x

    net = NeuralNetwork()

    lossfunction = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=.01)

    return net,lossfunction,optimizer

net,lossfunction,optimizer = create_neural_network()

X,y = iter(train_loader).next()
yHat = net(X)

def train_model():

    epochs = 50

    net,lossfunction,optimizer = create_neural_network()

    trainAcc = []

    for epoch in range(epochs):
        print(epoch)
        for X,y in train_loader:

            yHat = net(X)
            loss = lossfunction(yHat,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        X,y = next(iter(test_loader))
        yHat = net(X)
        trainAcc.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )

    return trainAcc,net
            
trainAcc,net = train_model()

plt.plot(trainAcc,'o')
plt.show()
