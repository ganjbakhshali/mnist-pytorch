import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F  # Add this line
import matplotlib.pyplot as plt  # Import matplotlib for plotting


# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
train_losses = []  # to store the training loss values during training
test_losses = []  # to store the test loss values during training

#Load model
model.load_state_dict(torch.load('mnist_cnn_model.pth'))
# for epoch in range(num_epochs):
    # model.train()
    # running_train_loss = 0.0
    # for images, labels in train_loader:
    #     images, labels = images.to(device), labels.to(device)

    #     optimizer.zero_grad()
    #     outputs = model(images)
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     running_train_loss += loss.item()

    # average_train_loss = running_train_loss / len(train_loader)
    # train_losses.append(average_train_loss)

    # print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss}")

    # # Evaluation on the test set
    # model.eval()
    # running_test_loss = 0.0
    # correct = 0
    # total = 0

    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)

    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         running_test_loss += loss.item()

    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # average_test_loss = running_test_loss / len(test_loader)
    # test_losses.append(average_test_loss)

    # accuracy = 100 * correct / total
    # print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {average_test_loss}, Accuracy: {accuracy}%")

# Save the trained model
# torch.save(model.state_dict(), 'mnist_cnn_model.pth')
# Plot the training and test curves
# plt.plot(train_losses, label='Training Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Curves')
# plt.legend()
# plt.savefig('training_test_curves.png')  # Save the figure as an image file
# plt.close()  # Close the plot to release resources
# Testing the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct / total}%")
