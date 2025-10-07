import torch
import torch.nn as nn
import torch.nn.functional as F

def train(train_loader, model, optimizer, device, i):

    train_loss_list = []
    final_output = []
    final_label = []

    # put the model in train mode
    model.train()

    # cross entropy loss
    criterion = nn.CrossEntropyLoss()

    j = 0

    for data in train_loader:
        j += 1

        if i == j:
            break
        
        print(data[0].shape, data[1].shape)
        print("\n")

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
        
        print(f"Loss :- {loss}")

         # zero grad the optimizer
        optimizer.zero_grad()

        # calculate the gradient
        loss.backward()

        # update the weights
        optimizer.step()

        
        #break
    

        """
        softmax_values = F.softmax(outputs, dim=1)
        outputs = torch.argmax(softmax_values, dim=1).int()

        OUTPUTS = outputs.detach().cpu().tolist()
        final_output.extend(OUTPUTS)
        final_label.extend(label.detach().cpu().tolist())

    return final_output, final_label, sum(train_loss_list)/len(train_loss_list)
"""

