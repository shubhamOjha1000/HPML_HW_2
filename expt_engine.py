import torch
import torch.nn as nn
import time

def train(train_loader, model, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    total_data_loading_time = 0.0
    total_training_time = 0.0
    epoch_start_time = time.perf_counter()

    # Start timing before first batch
    data_loading_start = time.perf_counter()

    for batch_idx, (features, labels) in enumerate(train_loader):
        # --- C2.1: Measure DataLoader fetch time ---
        data_loading_end = time.perf_counter()
        total_data_loading_time += (data_loading_end - data_loading_start)

        # --- C2.2: Training section (includes .to(device)) ---
        training_start = time.perf_counter()

        features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_end = time.perf_counter()
        total_training_time += (training_end - training_start)

        # --- Prepare timer for next batch’s DataLoader fetch ---
        data_loading_start = time.perf_counter()

        # Collect predictions
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.append(predictions.detach())
        all_labels.append(labels.detach())

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        
    epoch_end_time = time.perf_counter()
    total_epoch_time = epoch_end_time - epoch_start_time

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    return (all_predictions.cpu().tolist(),
            all_labels.cpu().tolist(),
            total_loss / len(train_loader),
            total_data_loading_time,   # C2.1
            total_training_time,       # C2.2
            total_epoch_time)          # C2.3
