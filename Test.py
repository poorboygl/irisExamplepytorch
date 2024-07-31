import numpy as np
import pandas as pd
import torch
from torchsummary import summary

from urllib.request import urlretrieve
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)
df = pd.read_csv(iris, sep=',')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes
print(df)
X = torch.tensor([[value for value in row] for row in df.values[:,0:4]],dtype = torch.float)
label_Y = df.values[:,4]
set_Y = set(df.values[:,4])
new_values = list(range(len(set_Y)))
replacement_map = dict(zip(set_Y, new_values))
Y = np.copy(label_Y)
for old_value, new_value in replacement_map.items():
    label_Y[label_Y == old_value] = new_value
Y = torch.tensor([value for value in label_Y[:]],dtype = torch.int64)

numberOfInputFeatures =  len(attributes)-1
numberOfOutputFeatures= len(set_Y)
model = torch.nn.Sequential(
     torch.nn.Linear(in_features=numberOfInputFeatures,out_features= numberOfOutputFeatures)
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#summary(model, input_size = (numberOfInputFeatures,))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() ,lr=0.01)
num_epoch = 10000
losses = []
for _  in range(num_epoch):  
    # Zero the gradients
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X).to(device)

    # Compute loss
    loss = criterion(outputs, Y.to(device))
    losses.append(loss.item())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

#print("Weight: ",model[0].weight)
#print("Bias:", model[0].bias)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.savefig('foo.png')
plt.savefig('foo.pdf')