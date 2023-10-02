import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from ResNets import *
from Normalization_Techniques import *
import os
import sys
f = open("BCN.txt", 'w')
sys.stdout = f
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    torch.manual_seed(0)


    # For CIFAR100 dataset
    # def split_cifar(data_dir):
    #  t = transforms.Compose([transforms.ToTensor(),])
    #  ts = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=t)
    #  loader = DataLoader(ts, batch_size=40000, shuffle=True, num_workers=0)

    def split_cifar(data_dir):
        t = transforms.Compose([transforms.ToTensor(), ])
        ts = torchvision.datasets.CIFAR10(root=data_dir, download=True, transform=t)
        loader = DataLoader(ts, batch_size=40000, shuffle=True, num_workers=0)
        a, b = [], []
        for _, (X, Y) in enumerate(loader):
            a.append(X)
            b.append(Y)
        return a[0], b[0], a[1], b[1], torch.mean(torch.cat((a[0], a[1]), dim=0),
                                                  dim=0)  # Xtrain, Ytrain, Xval, Yval, per_pixel_mean (3,32,32)


    # class Cifar100(Dataset):
    class Cifar10(Dataset):
        def __init__(self, X, Y, transform=None):
            super(Cifar10, self).__init__()
            # super(Cifar100, self).__init__()
            self.transform = transform
            self.X = X
            self.Y = Y

        def __getitem__(self, index):
            x, y = self.X[index], self.Y[index]
            x = self.transform(x)
            return x, y

        def __len__(self):
            return self.X.shape[0]


    def Save_Stats(trainacc, testacc, exp_name):
        data = []
        data.append(trainacc)
        data.append(testacc)
        data = np.array(data)
        data.reshape((2, -1))
        if not os.path.exists('./Results'):
            os.mkdir('./Results')
        stats_path = './Results/%s_accs.npy' % exp_name
        np.save(stats_path, data)
        SavePlots(data[0], data[1], 'Accuracy', exp_name)


    def SavePlots(y1, y2, metric, exp_name):
        try:
            plt.clf()
        except Exception as e:
            pass
        # plt.title(exp_name)
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        epochs = np.arange(1, len(y1) + 1, 1)
        plt.plot(epochs, y1, label='Train %s' % metric)
        plt.plot(epochs, y2, label='Validation %s' % metric)
        ep = np.argmax(y2)
        plt.legend()
        plt.savefig('./Results/%s_%s' % (exp_name, metric), dpi=300)
    parser = argparse.ArgumentParser(description='Training ResNets with Normalizations on CIFAR10 or CIFAR100')
    parser.add_argument('--num_epochs', type=int, default=90)
    parser.add_argument('--normalization', default='BCN', type=str)
    # parser.add_argument('--data_dir',default='./Dataset/cifar-100-batches-py', type=str)
    parser.add_argument('--data_dir', default='./Dataset/cifar-100-batches-py', type=str)
    parser.add_argument('--output_file', default='./Output_Result/1.1.pth', type=str)
    parser.add_argument('--n', default=2, help='Number of (Per) residual blocks', type=int)
    parser.add_argument('--r', default=10,
                        help='Number of classes 10 for CIFAR10 dataset and 100 classes for CIFAR100 dataset', type=int)
    parser.add_argument('--save_plots', default=True, help='Give argument if plots to be plotted')
    args = parser.parse_args()

    print(args.normalization)

    n = args.n
    r = args.r
    normalization_layers = {'Torch_BN': nn.BatchNorm2d, 'BN': BatchNorm2D, 'IN': InstanceNorm2D, 'LN': LayerNorm2D,
                            'GN': GroupNorm2D, 'BCN': BatchChannelNorm, 'NN': None}
    norm_layer_name = args.normalization
    norm_layer = normalization_layers[norm_layer_name]

    # create the required ResNet model
    model = ResNet(n, r, norm_layer_name, norm_layer)
    # print(model(torch.rand((2,3,32,32))).shape) #: all resnet models working [CHECKED]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # create train-val-test split of CIFAR10
    X_train, Y_train, X_val, Y_val, per_pixel_mean = split_cifar(data_dir=args.data_dir[:-19])


    def get_transforms(train=True):
        train_transforms = [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (1., 1., 1.)), ]
        val_transforms = [transforms.ToPILImage(), transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (1., 1., 1.)), ]
        return train_transforms if train else val_transforms


    train_transform = transforms.Compose(get_transforms(train=True))
    val_transform = transforms.Compose(get_transforms(train=False))
    # trainset = Cifar100(X_train, Y_train, transform=train_transform)
    trainset = Cifar10(X_train, Y_train, transform=train_transform)

    trainset_loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    # valset = Cifar100(X_val, Y_val, transform=val_transform)
    valset = Cifar10(X_val, Y_val, transform=val_transform)
    valset_loader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=0)

    start_epoch, end_epoch = 1, args.num_epochs + 1
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_acc, val_acc = [], []
    for epoch in range(start_epoch, end_epoch):
        if epoch == 75 or epoch == 85:
            lr /= 10.
            optimizer = optim.SGD(model.parameters(), lr=lr)
        # Training
        model.train()
        total_samples, correct_predictions = 0, 0
        for _, (X, Y) in enumerate(trainset_loader):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()  # remove history
            Y_ = model(X)
            Y_predicted = Y_.argmax(dim=1)

            loss = loss_fn(Y_, Y)
            loss.backward()
            optimizer.step()  # update weights/biases

            # __, Y_predicted = Y_.max(1)
            correct_prediction = Y_predicted.eq(Y).sum()
            correct_predictions += correct_prediction.item()
            total_samples += Y_predicted.size(0)
        train_acc.append((correct_predictions / total_samples) * 100.)
       # np.savetxt('Training_Accuracy.txt',train_acc)

        # Testing
        model.eval()  # this is useful in informing nn.modules to appropriately behave during inference (for example: nn.Dropout)
        total_samples, correct_predictions = 0, 0
        with torch.no_grad():
            for _, (X, Y) in enumerate(valset_loader):
                X = X.to(device)
                Y = Y.to(device)
                Y_ = model(X)
                Y_predicted = Y_.argmax(dim=1)
                loss = loss_fn(Y_, Y)

                # __, Y_predicted = Y_.max(1)
                correct_prediction = Y_predicted.eq(Y).sum()
                correct_predictions += correct_prediction.item()
                total_samples += Y_predicted.size(0)
        val_acc.append((correct_predictions / total_samples) * 100.)
        print("Epoch ", epoch, "Training  Accuracy", train_acc[-1], "Validation Accuracy", val_acc[-1])
       # np.savetxt('Validation_Accuracy.txt',val_acc)
       # Training and Testing completed. Save the model Parameters and Plots
    torch.save(model.state_dict(), args.output_file)

    if args.save_plots != False:
        Save_Stats(train_acc, val_acc, args.normalization)
f.close