from sportsreference.mlb.teams import Teams
from sportsreference.mlb.roster import Roster
from sportsreference.mlb.roster import Player
import torch
import torch.nn as nn
import torch.optim as optim
import csv

class model(nn.Module):
    def __init__(self, input, output):
        super(model, self).__init__()
        self.fc1 = nn.Linear(input, output)

    def forward(self, x):
        output = self.fc1(x)
        return output

def get_accuracy(model, data_x, data_y, n):
    sum = 0
    for i in range(len(data_x)):
        pred = model(torch.tensor([data_x[i]])).item()
        if (data_y[i] - n) < pred < (data_y[i] + n):
            sum += 1
    return round(sum / len(data_x), 4)

train_x = []
train_y = []

with open ('2019.csv','r', encoding='utf8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        if float(row[16]) > 0:
            train_x.append(float(row[16]))
            train_y.append(float(row[8]) / (162 / int(row[1])))

split = int(len(train_x) * 0.8)
test_x = train_x[split:]
train_x = train_x[0:split]
test_y = train_y[split:]
train_y = train_y[0:split]

train_x_tensor = torch.tensor(train_x)
train_x_tensor = train_x_tensor.reshape((list(train_x_tensor.size())[0],1))
train_y_tensor = torch.tensor(train_y)
train_y_tensor = train_y_tensor.reshape((list(train_x_tensor.size())[0],1))

model = model(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(801):
    optimizer.zero_grad()
    outputs = model(train_x_tensor)
    loss = criterion(outputs, train_y_tensor)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

print("Train accuracy is {} within 5 HRs and {} within 10 HRs".format(get_accuracy(model, train_x, train_y, 5), get_accuracy(model, train_x, train_y, 10)))
print("Test accuracy is {} within 5 HRs and {} within 10 HRs".format(get_accuracy(model, test_x, test_y, 5), get_accuracy(model, test_x, test_y, 10)))