from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.onnx
import random
import json

# remove extra EZKL formatting
# with open('input.json', 'r') as f:
#     wrapped_data = json.load(f)

# # Get the data that was wrapped within "input_data"
# json_data = wrapped_data["input_data"]

# # Write the unwrapped data back to the JSON file
# with open('input2.json', 'w') as f:
#     json.dump(json_data, f)

# Loading the data - creates vulnerability for verifying input data, but shortcut needed to circumvent
    # 'failed to deserialize Datasource' error
df = pd.read_json('cleandata.json')

# obviously, reducing determinism is a terribly wrong way to classify in any model - but it's funny.
def classify(row):
    
    time = datetime.fromtimestamp(row['Time']/1e3) #.strptime(row['Time'], '%Y-%m-%d %H:%M:%S')
    
    if row['Token'] == 'SAFEMOON' and time > datetime(2022, 12, 14):
        if random.uniform(0, 1) < 0.8:  # 80% chance of being 'fish'
            return 0
    elif row['Token'] == 'HNT':
        if random.uniform(0, 1) < 0.7:  # 70% chance of being 'neither'
            return 1
    elif row['Token'] == 'ETH' and time > datetime(2022, 12, 24):
        if random.uniform(0, 1) < 0.7:  # 70% chance of being 'shark'
            return 2
    elif row['Token'] == 'PEPE' and time > datetime(2023, 5, 3):
        if random.uniform(0, 1) < 0.8:  # 80% chance of being 'shark'
            return 2
    elif row['Token'] == 'APTOS':
        if random.uniform(0, 1) < 0.6:  # 60% chance of being 'fish'
            return 0
    elif row['Token'] == 'MATIC' and time > datetime(2022, 6, 18):
        if random.uniform(0, 1) < 0.7:  # 70% chance of being 'neither'
            return 1
    
    return 1  # Default to 'neither' if none of the conditions are met or the random number is outside the desired range

df['Classification'] = df.apply(classify, axis=1)

# Perform one-hot encoding on 'Trade' and 'Token' columns
df = pd.get_dummies(df, columns=['Trade', 'Token'])
print(df)

# Drop 'Time' and 'Trader' columns as they are not needed for the model
df.drop(columns=['Time', 'User'], inplace=True)

# Define input and output columns
X = df.drop('Classification', axis=1).values
y = df['Classification'].values
print(X)
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Define the architecture of the model
class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and define loss and optimizer
model = Circuit()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(500):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Evaluate the model
y_pred_test = model(X_test)
_, predicted = torch.max(y_pred_test, 1)

# Output classification report
print(classification_report(y_test, predicted))

# Save the trained model
# torch.onnx.export(model, X_test, 'network2.onnx')

torch.onnx.export(model,               # model being run
                      # model input (or a tuple for multiple inputs)
                      X,
                      # where to save the model (can be a file or file-like object)
                      "network.onnx",
                      export_params=False,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})

data = dict(    input_data=[X])

# Serialize data into file:
json.dump(data, open("input.json", 'w'))

# # Preprocessing the data
# for col in ['User', 'Trade', 'Token']:
#     df[col] = LabelEncoder().fit_transform(df[col])

# X = df.drop('Skill', axis=1).values
# y = LabelEncoder().fit_transform(df['Skill'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Building the model
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(X.shape[1], 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 3)
#         )

#     def forward(self, x):
#         return self.layers(x)

# model = MLP()

# # Training the model
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# X_train = torch.FloatTensor(X_train)
# y_train = torch.LongTensor(y_train)

# for epoch in range(100):  # number of epochs
#     optimizer.zero_grad()
#     output = model(X_train)
#     loss = criterion(output, y_train)
#     loss.backward()
#     optimizer.step()

# # Saving the model to an ONNX file
# dummy_input = torch.randn(1, X.shape[1])
# torch.onnx.export(model, dummy_input, "model.onnx")
