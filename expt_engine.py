import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def train(train_loader, model, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    # Initialize timing variables
    total_data_loading_time = 0.0
    total_training_time = 0.0
    epoch_start_time = time.perf_counter()
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Start data loading timer
        data_loading_start = time.perf_counter()
        
        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # End data loading timer
        data_loading_end = time.perf_counter()
        total_data_loading_time += (data_loading_end - data_loading_start)
        
        # Start training timer
        training_start = time.perf_counter()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # End training timer
        if device.type == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        training_end = time.perf_counter()
        total_training_time += (training_end - training_start)
        
        # Collect predictions
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.append(predictions.detach())
        all_labels.append(labels.detach())
        
        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Calculate total epoch time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_end_time = time.perf_counter()
    total_epoch_time = epoch_end_time - epoch_start_time
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    return (all_predictions.cpu().tolist(), 
            all_labels.cpu().tolist(), 
            total_loss / len(train_loader),
            total_data_loading_time,    # C2.1: Data-loading time
            total_training_time,        # C2.2: Training time  
            total_epoch_time)           # C2.3: Total epoch time