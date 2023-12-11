import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show tensor values for the first image in the batch
print(images[0])

# Define a simple neural network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Train the network
for epoch in range(2):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

def visualize_weights(layer):
    with torch.no_grad():
        weights = layer.weight.data
        #weights = weights.view(-1, 28, 28)  # Reshape for MNIST 28x28 images

        fig, axes = plt.subplots(10, 10, figsize=(10, 10))  # Adjust the subplot layout and size
        for i, ax in enumerate(axes.flat):
            if i < weights.shape[0]:
                ax.imshow(weights[i], cmap='gray')
                ax.axis('off')
        plt.show()

visualize_weights(net.fc1)

# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
