import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def torch_test():
    device = torch.device('cuda') # or 'cpu'

    transform = transforms.Compose(
    [transforms.ToTensor()]) # here we could put a list of operations
    batch_size=32
    trainset = torchvision.datasets.mnist.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

    testset = torchvision.datasets.mnist.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters())
    num_mini=200
    acc_hist = []
    for epoch in range(2):  # loop over the dataset multiple times
        correct= 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs.to(device))
            # loss
            loss = criterion(outputs, labels.to(device))
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # some status output
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.to(device).data.view_as(pred)).sum()

            running_loss += loss.item()
            if i % num_mini == num_mini-1:    # print every 200 mini-batches
                acc = correct.to('cpu').numpy()/200/batch_size
                acc_hist.append(acc)
                print('[%d, %5d] loss: %.3f, accuracy: %.3f' %
                      (epoch + 1, i + 1,running_loss / num_mini, acc))
                running_loss = 0.0
                correct = 0

    print('Finished Training')
    plt.plot(acc_hist)
    plt.show()

    testlosses=[]
    # When using the model after training we want it's batch norm and dropout layers work in infrerence mode, not training one.
    net.to('cpu').eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # we don't need to waste time for estimation of the gradients
        for data, target in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    testlosses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)




if __name__ == '__main__':
    torch_test()