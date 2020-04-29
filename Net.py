import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pandas as pd
import numpy as np
import math

class model(nn.Module):
    def __init__(self, input, hidden, output):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        z = self.fc1(x)
        output = self.fc2(z)
        return output

def get_accuracy(model, data_x, data_y, n):
    sum = 0
    for i in range(len(data_x)):
        pred = model(torch.tensor([data_x[i]])).item()
        if (data_y[i] - n) < pred < (data_y[i] + n):
            sum += 1
    return round(sum / len(data_x), 4)

def get_loss(model, X, y, crit):
    outputs = model(X)
    outputs = outputs.view(list(outputs.size())[0])
    loss = criterion(outputs, y)
    return loss

data = pd.read_csv("2019.csv")
X = data.iloc[:,25:28]
y = data.iloc[:,-1]

best_features = SelectKBest(f_regression, k = 3)
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
feature_scores = pd.concat([df_columns, df_scores], axis = 1)
feature_scores.columns = ["Feature", "Score"]
# print(feature_scores.nlargest(3, "Score"))

train_x = []
train_y = []

with open ('2019.csv','r', encoding='utf8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        train_x.append((float(row[25]), float(row[26]), float(row[27])))
        train_y.append(float(row[-1]))

split = int(len(train_x) * 0.8)
test_x = train_x[split:]
train_x = train_x[0:split]
test_y = train_y[split:]
train_y = train_y[0:split]

train_x_tensor = torch.tensor(train_x)
# train_x_tensor = train_x_tensor.reshape((list(train_x_tensor.size())[0],1))
train_y_tensor = torch.tensor(train_y)
# train_y_tensor = train_y_tensor.reshape((list(train_x_tensor.size())[0],1))

model = model(3, 2, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for epoch in range(1001):
    optimizer.zero_grad()
    loss = get_loss(model, train_x_tensor, train_y_tensor, criterion)
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

# print("Train accuracy is {} within 5 HRs and {} within 10 HRs".format(get_accuracy(model, train_x, train_y, 5), get_accuracy(model, train_x, train_y, 10)))
# print("Test accuracy is {} within 5 HRs and {} within 10 HRs".format(get_accuracy(model, test_x, test_y, 5), get_accuracy(model, test_x, test_y, 10)))

test_x_tensor = torch.tensor(test_x)
test_y_tensor = torch.tensor(test_y)

train_loss = get_loss(model, train_x_tensor, train_y_tensor, criterion)
test_loss = get_loss(model, test_x_tensor, test_y_tensor, criterion)

print("Model is +- {} HRs on the training data".format(round(math.sqrt(train_loss.item()), 2)))
print("Model is +- {} HRs on the testing data".format(round(math.sqrt(test_loss.item()), 2)))