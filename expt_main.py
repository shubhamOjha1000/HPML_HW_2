import argparse
from download_CIFAR10_training_data import download_cifar10
from model import ResNet18, ResNet18_NoBN
import torch
from torch.utils.data import DataLoader
from dataset import CIFAR10_dataset
import engine 
import expt_engine
import torch.optim as optim
from utils import Accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of total training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=2, type=int, help='num workers for dataloader')
    parser.add_argument('--optimizer', default='sgd', type=str, help='choose optimize')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate for optimizer')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
    parser.add_argument('--train_data_path', default='/content', type=str, help='path to the cifar10 training data')
    parser.add_argument('--num_classes', default=10, type=int, help='2:- No of classes in cifar10 dataset')
    parser.add_argument('--model', default='RESNET18', type=str, help='Model to be used')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--Q3', default=False, type=bool, help='How much to run Q3')
    parser.add_argument('--Q4', default=False, type=bool, help='How much to run Q4')
    args = parser.parse_args()

    # download cifar10 training data
    cifar10_training_data = download_cifar10(args.train_data_path)

    # cifar10 Dataset
    training_data = CIFAR10_dataset(cifar10_training_data, img_transform=True)

    # Dataloader
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')



    # model
    if args.model == 'RESNET18':
        model = ResNet18(args.num_classes)
        model.to(device)

    elif args.model == 'ResNet18_NoBN':
        model = ResNet18_NoBN(args.num_classes)
        model.to(device)


    # --- Set up the optimizer ---
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optimizer.lower() == 'sgd_nesterov':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer '{args.optimizer}'. Please choose from: "
                        "'SGD', 'SGD_Nesterov', 'Adagrad', 'Adadelta', 'Adam'")
    



    
        # --- Q3 & Q4: Parameter and Gradient Counting ---
        # --- Q3 & Q4: Parameter and Gradient Counting ---
    if args.Q3 or args.Q4:
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_gradients = total_trainable_params  # one gradient per scalar parameter

        print("\n=== MODEL PARAMETER & GRADIENT INFO ===")
        print(f"Optimizer: {args.optimizer.upper()}")
        print(f"Total trainable parameters (scalars): {total_trainable_params:,}")
        print(f"Total gradients (scalars): {total_gradients:,}")
        print("----------------------------------------\n")

        # Exit after printing for Q3/Q4
        return






    # Lists to store timing results for each epoch
    data_loading_times = []
    training_times = []
    total_epoch_times = []

    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        
        final_output, final_label, training_loss, data_loading_time, training_time, total_epoch_time = expt_engine.train(
            train_loader, model, optimizer, device
        )

        top1_training_accuracy = Accuracy(final_label, final_output)

        # Store timing results
        data_loading_times.append(data_loading_time)
        training_times.append(training_time)
        total_epoch_times.append(total_epoch_time)

        print(f'Training Loss: {training_loss:.4f}')
        print(f'Top-1 Training Accuracy: {top1_training_accuracy:.4f}')
        print(f'Data-loading Time: {data_loading_time:.2f} seconds')      # C2.1
        print(f'Training Time: {training_time:.2f} seconds')              # C2.2
        print(f'Total Epoch Time: {total_epoch_time:.2f} seconds')        # C2.3
        print('-' * 50)

    # Print summary after all epochs
    print('\n=== TIMING SUMMARY ===')
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch+1}:')
        print(f'  Data-loading: {data_loading_times[epoch]:.2f}s')
        print(f'  Training: {training_times[epoch]:.2f}s')
        print(f'  Total: {total_epoch_times[epoch]:.2f}s')
    
    print(f'\nAverages over {args.num_epochs} epochs:')
    print(f'  Data-loading: {sum(data_loading_times)/len(data_loading_times):.2f}s')
    print(f'  Training: {sum(training_times)/len(training_times):.2f}s')
    print(f'  Total: {sum(total_epoch_times)/len(total_epoch_times):.2f}s')

if __name__ == '__main__':
    main()