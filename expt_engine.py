import torch
import torch.nn as nn
import torch.nn.functional as F

def train(train_loader, model, optimizer, device, i):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Collect predictions (on GPU, move to CPU at end)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.append(predictions.detach())
        all_labels.append(labels.detach())
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Concatenate all results (on GPU first)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    return (all_predictions.cpu().tolist(), 
            all_labels.cpu().tolist(), 
            total_loss / len(train_loader))