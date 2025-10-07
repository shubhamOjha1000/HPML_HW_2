import argparse
from download_CIFAR10_training_data import download_cifar10
from model import ResNet18
import torch
from torch.utils.data import DataLoader
from dataset import CIFAR10_dataset
import engine 
import torch.optim as optim
from utils import Accuracy

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=5, type=int, help= 'Number of total training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=2, type=int, help='num workers for dataloader')
    parser.add_argument('--optimizer', default='SGD', type=str, help='choose between: SGD or Adam')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate for optimizer')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
    parser.add_argument('--train_data_path', default='/content', type=str, help='path to the cifar10 training data')
    parser.add_argument('--num_classes', default=10, type=int, help='2:- No of classes in cifar10 dataset')
    parser.add_argument('--model', default='RESNET18', type=str, help='Model to be used')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--i', default=1, type=int, help='How much to run')
    args = parser.parse_args()

    # download cifar10 training data
    cifar10_training_data = download_cifar10(args.train_data_path)

    #cifar10 Dataset
    training_data = CIFAR10_dataset(cifar10_training_data, img_transform=True)

    #Dataloader
    train_loader = DataLoader(training_data, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda use')
    else:
        device = torch.device('cpu')
        print('cpu use')

    # model :- 
    if args.model == 'RESNET18':
        model = ResNet18(args.num_classes)
        model.to(device)
        

    # set up the optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)



    for epoch in range(args.num_epochs):
        print(f'epoch :-{epoch}')

        engine.train(train_loader, model, optimizer, device, args.i)
        print(f'trainig completed')
        """
        final_output, final_label, training_loss = engine.train(train_loader, model, optimizer, device)

        top1_training_accuracy = Accuracy(final_label, final_output)

        print(f'training loss :- {training_loss}')
        print(f'top1 training accuracy :- {top1_training_accuracy}')
        """





if __name__ == '__main__':
    main()