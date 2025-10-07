import torch
import torch.nn as nn
import torch.nn.functional as F

def train(train_loader, model, optimizer, device):

    train_loss_list = []
    final_output = []
    final_label = []

    # put the model in train mode
    model.train()

    # cross entropy loss
    criterion = nn.CrossEntropyLoss()

    for data in train_loader:
        feature = data[0]
        label = data[1]

        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)
        label = label.to(device)

        # do the forward pass through the model
        outputs = model(feature)

        # calculate loss
        loss = criterion(outputs, label)
        train_loss_list.append(loss)

         # zero grad the optimizer
        optimizer.zero_grad()

        # calculate the gradient
        loss.backward()

        # update the weights
        optimizer.step()

        softmax_values = F.softmax(outputs, dim=1)
        outputs = torch.argmax(softmax_values, dim=1).int()

        OUTPUTS = outputs.detach().cpu().tolist()
        final_output.extend(OUTPUTS)
        final_label.extend(label.detach().cpu().tolist())

    return final_output, final_label, sum(train_loss_list)/len(train_loss_list)

