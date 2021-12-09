import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_set = datasets.MNIST('./data', train=training, download=True, transform=custom_transform)
    loader = torch.utils.data.DataLoader(data_set, batch_size=50)
    return loader


def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        accuracy = 0
        total = 0

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += 50 * loss.item()

            predicted = torch.max(outputs.data, 1)[1]
            accuracy += (predicted == labels).sum().item()
            total += labels.size(0)

        percent = '%.2f' % (accuracy * 100 / 60000)
        loss = '%.3f' % (running_loss / 60000)
        print(f'Train Epoch: {epoch}   Accuracy: {accuracy}/60000({percent}%)  Loss: {loss}')


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    total = 0
    correct = 0
    valid_loss = 0.0
    model.eval()

    with torch.no_grad():

        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            predictions = torch.max(outputs, 1)[1]
            valid_loss += 50 * criterion(outputs, labels).item()

            for prediction, label in zip(predictions, labels):
                if prediction == label:
                    correct += 1
                total += 1

    if show_loss:
        average_loss = '%.4f' % (valid_loss / len(test_loader.sampler))
        print(f'Average loss: {average_loss}')

    accuracy = '%.2f' % (100 * correct / total)
    print(f'Accuracy: {accuracy}%')


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    with torch.no_grad():
        model.eval()
        outputs = model(test_images[index])
        prob = F.softmax(outputs, dim=1)

        values, indices = torch.topk(prob, 3)
        values = values.tolist()[0]
        indices = indices.tolist()[0]

        for i in range(3):
            prediction = class_names[indices[i]]
            value = '%.2f' % (values[i]*100)
            print(f'{prediction}: {value}%')


if __name__ == '__main__':
    #  test get_data_loader
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print()

    #  test build_loader
    model = build_model()
    print(model)
    print()

    #  test train_model
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)
    print()

    #  test evaluate_model
    evaluate_model(model, test_loader, criterion, show_loss=True)
    print()

    #  test predict_label
    pred_set = []
    for i, (images, label) in enumerate(test_loader.dataset):
        pred_set.append(images)
    pred_set = torch.stack(pred_set)
    predict_label(model, pred_set, 1)
