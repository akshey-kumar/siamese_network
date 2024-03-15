import torch
from torch.nn.functional import pairwise_distance


def predict_label(output1, output2, margin):
    euclidean_distance = pairwise_distance(output1, output2, keepdim=True)
    label_pred = euclidean_distance > margin
    return label_pred


def train_loop(dataloader, device, net, loss_fn, optimizer):
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


def test_loop(dataloader, device, net, loss_fn):
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
