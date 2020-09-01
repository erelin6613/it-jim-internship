import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class Net_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(20)
        # self.conv2_drop = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(20, 50, kernel_size=3)
        self.conv4 = nn.Conv2d(50, 100, kernel_size=3)
        self.conv4_drop = nn.Dropout2d(p=0.3)
        self.norm2 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(1600, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2(self.conv1(x)), 2))
        x = self.norm1(x)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(self.conv1(x))), 2))
        # print(x.shape)
        # x = F.relu(F.max_pool2d(self.conv4(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_drop(self.conv4(self.conv3(x))), 2))
        x = self.norm2(x)
        # print(x.shape)
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.fc3(x)
        return F.log_softmax(x)


def testing(net):
    batch_size = 128

    # transformation for the test set should not contain augmentation.
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.mnist.MNIST(root='./data', train=False,
                                               download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    correct = 0
    total = 0
    predict_label = []
    test_label = []
    with torch.no_grad():
        for data, label in testloader:
            test_label.extend(label)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            predict_label.extend(predicted.tolist())
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('accuracy {:.2f}%'.format(100 * correct / total))

    precision, recall, fbeta, sup = precision_recall_fscore_support(test_label, predict_label, average='micro')
    print('precision {:.3f}\trecall {:.3f}'.format(precision, recall))
    print(confusion_matrix(test_label, predict_label))


def training(net):
    # augmentation -- randomly selects a rectangle region in an image and erases its pixels
    transform = transforms.Compose([transforms.RandomRotation(degrees=(90, 90)), transforms.ToTensor(),
                                    transforms.RandomErasing(p=0.5, scale=(0.04, 0.2),
                                                             value=1),
                                    transforms.Normalize((0.5,), (0.5,))])

    batch_size = 128
    trainset = torchvision.datasets.mnist.MNIST(root='./data', train=True,
                                                download=True, transform=transform)

    # splitting a small portion of the data for validation
    val_samples = 1024
    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - val_samples, val_samples])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    validloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                              shuffle=True, num_workers=1)

    # loss function
    loss_func = F.cross_entropy

    # optimization algorithm
    opt = optim.Adam(net.parameters(), lr=0.001)

    # for loss visualisation
    loss_graph = []

    for epoch in range(2):
        net.train()
        # over each mini-batch
        for data, label in trainloader:
            # predict using a simple linear model
            predict = net(data)

            # calculate loss
            loss = loss_func(predict, label)

            # update the gradients of the model -- weights and bias here
            loss.backward()

            # update parameters and reset the gradient to 0
            opt.step()
            opt.zero_grad()

        net.eval()
        with torch.no_grad():
            val_loss = sum(loss_func(net(data), label) for data, label in validloader)

        val_loss = val_loss / len(validloader)
        loss_graph.append(val_loss)
        print('epoch: {:1d}, loss: {:.3f}'.format(epoch, val_loss))

    # save the model
    torch.save(net.state_dict(), 'model.pth')


def transfer_learn(net):

    # # freeze the entire model except for the last layer and retrain only it
    # for name, child in net.named_children():
    #     if name == 'fc3':
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False

    # # freeze all convolutional layers that extract features and retrain a classifier part
    # for name, child in net.named_children():
    #     if name in ['fc1', 'fc2', 'fc3']:
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False

    # augmentation -- randomly selects a rectangle region in an image and erases its pixels
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomErasing(p=0.5, scale=(0.04, 0.2),
                                                             value=1),
                                    transforms.Normalize((0.5,), (0.5,))])

    batch_size = 128
    trainset = torchvision.datasets.mnist.MNIST(root='./data', train=True,
                                                download=True, transform=transform)

    # splitting a small portion of the data for validation
    val_samples = 1024
    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - val_samples, val_samples])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    validloader = torch.utils.data.DataLoader(valset, batch_size=1024,
                                              shuffle=True, num_workers=1)

    # loss function
    loss_func = F.cross_entropy

    # optimization algorithm
    opt = optim.Adam(net.parameters(), lr=0.001)

    # for loss visualisation
    loss_graph = []

    for epoch in range(10):
        net.train()
        # over each mini-batch
        for data, label in trainloader:
            # predict using a simple linear model
            predict = net(data)

            # calculate loss
            loss = loss_func(predict, label)

            # update the gradients of the model -- weights and bias here
            loss.backward()

            # update parameters and reset the gradient to 0
            opt.step()
            opt.zero_grad()

        net.eval()
        with torch.no_grad():
            val_loss = sum(loss_func(net(data), label) for data, label in validloader)

        val_loss = val_loss / len(validloader)
        loss_graph.append(val_loss)
        print('epoch: {:1d}, loss: {:.3f}'.format(epoch, val_loss))


if __name__ == '__main__':
    net = Net_CNN()

    # for name, child in net.named_children():
    #     print(name)

    net.load_state_dict(torch.load('model.pth'))
    transfer_learn(net)
    testing(net)
