# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(in_features=400, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=128)
        self.linear_3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        shape_dict = {}
        temp = self.conv_1(x)
        temp = self.relu(temp)
        stage_1 = self.max_pool(temp)
        shape_dict[1] = stage_1.size()

        temp = self.conv_2(stage_1)
        temp = self.relu(temp)
        stage_2 = self.max_pool(temp)
        shape_dict[2] = stage_2.size()

        stage_3 = self.flatten(stage_2)
        shape_dict[3] = stage_3.size()

        temp = self.linear_1(stage_3)
        stage_4 = self.relu(temp)
        shape_dict[4] = stage_4.size()

        temp = self.linear_2(stage_4)
        stage_5 = self.relu(temp)
        shape_dict[5] = stage_5.size()

        stage_6 = self.linear_3(stage_5)
        shape_dict[6] = stage_6.size()

        out = stage_6

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for parameter in model.parameters():
        if parameter.requires_grad:
            model_params += parameter.numel()

    model_params = model_params/1000000

    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
