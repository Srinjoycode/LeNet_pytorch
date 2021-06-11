"""
An implementation of LeNet CNN architecture.

"""

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Lambda
from torch import optim  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# defining the LeNet model


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=(5, 5), stride=(1, 1),
                               padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


model = LeNet(in_channels=in_channels, num_classes=num_classes).to(device)
print(model)

# Loading the MNIST datasets
tranform = torchvision.transforms.Compose([ToTensor(), torchvision.transforms.Pad(2)])

train_ds = torchvision.datasets.MNIST(root='./data',
                                      download=True,
                                      train=True,
                                      transform=tranform,
                                      )
test_ds = torchvision.datasets.MNIST(root='./data',
                                     download=True,
                                     train=False,
                                     transform=tranform,
                                     )
train_data_loader = torch.utils.data.DataLoader(train_ds,
                                                batch_size=batch_size,
                                                shuffle=True,

                                                )

test_data_loader = torch.utils.data.DataLoader(test_ds,
                                               batch_size=batch_size,
                                               shuffle=True)

train_features, train_labels = next(iter(train_data_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training the model
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_data_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_data_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_data_loader, model) * 100:.2f}")
