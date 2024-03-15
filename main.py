import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from siamese_network import SiameseNetwork, ContrastiveLoss, SiameseNetworkDataset


class Config():
    data_dir = "../data/western_blots"
    train_batch_size = 128
    train_number_epochs = 25


# Chance performance
labels = np.loadtxt(Config.data_dir + '/labels').astype(int)
chance_accuracy = []
for i in range(100):
    labels_perm = labels[np.random.permutation(labels.size)]
    chance_accuracy.append( (labels == labels_perm).sum()/labels.size )

print("Chance accuracy: ", np.mean(chance_accuracy), " pm ", np.std(chance_accuracy))


# Creating train and test datasets
dataset_train = SiameseNetworkDataset(
    data_dir=Config.data_dir,
    data_indices='train',
    should_invert=False,
    transform=transforms.Compose([transforms.Resize((100, 100)),
                                  transforms.ToTensor()
                                  ]))

dataset_test = SiameseNetworkDataset(
    data_dir=Config.data_dir,
    data_indices='test',
    should_invert=False,
    transform=transforms.Compose([transforms.Resize((100, 100)),
                                  transforms.ToTensor()
                                  ]))

train_dataloader = DataLoader(dataset_train,
                              shuffle=True,
                              num_workers=4,
                              batch_size=Config.train_batch_size)

test_dataloader = DataLoader(dataset_test,
                             shuffle=True,
                             num_workers=4,
                             batch_size=Config.train_batch_size)


def predict_label(output1, output2, margin):
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    label_pred = euclidean_distance > margin
    return label_pred


def train_loop(dataloader, net, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss, accuracy = 0, 0
    # Set the net to training mode - important for batch normalization and dropout layers
    net.train()
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        img0, img1, label = [x.to(device) for x in data]
        output1, output2 = net(img0, img1)
        loss = loss_fn(output1, output2, label)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item()

        # accuracy
        pred = predict_label(output1, output2, margin=loss_fn.margin)
        accuracy += (pred == label).type(torch.float).sum().item()

    avg_loss /= num_batches
    accuracy /= size
    print(f"Train Error: Avg loss: {avg_loss:>8f} Accuracy: {accuracy:>0.3f}\n")

    return avg_loss, accuracy


def test_loop(dataloader, net, loss_fn):
    # Set the net to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    net.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss, accuracy = 0, 0

    # Evaluating the net with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataloader:
            img0, img1, label = [x.to(device) for x in data]
            output1, output2 = net(img0, img1)

            avg_loss += loss_fn(output1, output2, label).item()

            pred = predict_label(output1, output2, margin=loss_fn.margin)
            accuracy += (pred == label).type(torch.float).sum().item()

    avg_loss /= num_batches
    accuracy /= size
    print(f"Test Error: Avg loss: {avg_loss:>8f} Accuracy: {accuracy:>0.3f}\n")

    return avg_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SiameseNetwork().to(device)
loss_fn = ContrastiveLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

train_history = []
test_history = []
for epoch in range(Config.train_number_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_loss = train_loop(train_dataloader, net, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, net, loss_fn)
    train_history.append(train_loss)
    test_history.append(test_loss)

train_history = np.array(train_history)
test_history = np.array(test_history)
print("Done!")

# Plotting
plt.figure()
plt.plot(train_history[:,1], label="Train accuracy")
plt.plot(test_history[:,1], label="Test accuracy", linestyle="--")
plt.hlines(y=np.max(chance_accuracy), xmin=0, xmax=Config.train_number_epochs-1,   )
plt.legend()
plt.show()

