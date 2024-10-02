from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from misc_utils import *
from models.resnet import *
from models.wideresnet import *
from trades import trades_loss

parser = argparse.ArgumentParser(
    description="PyTorch CIFAR TRADES Adversarial Training"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=200,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
parser.add_argument(
    "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
)
parser.add_argument("--weight-decay", "--wd", default=2e-4, type=float, metavar="W")
parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate")
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument("--epsilon", default=0.031, help="perturbation")
parser.add_argument("--num-steps", default=10, help="perturb number of steps")
parser.add_argument("--step-size", default=0.007, help="perturb step size")
parser.add_argument(
    "--beta", default=6.0, help="regularization, i.e., 1/lambda in TRADES"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--model-dir",
    default="logs/trades/cifar10/model-cifar10-wideResNet-v01",
    help="directory of model for saving checkpoint",
)

parser.add_argument("--num_class", default=10, type=int)
# Data
parser.add_argument("--data_name", default="cifar10")

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
print("###### DATASET: ", args.data_name)
if args.data_name == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root="data/cifar100", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="data/cifar100", train=False, download=True, transform=transform_test
    )
else:
    trainset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=transform_test
    )

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
)


def save_checkpoint(
    epoch,
    model,
    optimizer,
    ALL_TRAIN_LOSS,
    ALL_VAL_ACCU,
    args,
    verbose=True,
):
    model_dir = args.model_dir
    if not os.path.exists("%s" % (model_dir)):
        os.makedirs("%s" % (model_dir))
    TRAIN_STATE_FILE = str("%s/last_train_checkpoint.pth" % (model_dir))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "list_train_loss": ALL_TRAIN_LOSS,
        "list_val_accuracy": ALL_VAL_ACCU,
    }

    torch.save(checkpoint, TRAIN_STATE_FILE)

    if verbose:
        print(f"Checkpoint saved for epoch {epoch}.")


def load_checkpoint(
    model,
    optimizer,
    model_dir,
    verbose=True,
):
    filename = str("%s/last_train_checkpoint.pth" % (model_dir))

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    all_train_loss = checkpoint["list_train_loss"]
    all_val_accu = checkpoint["list_val_accuracy"]

    if verbose:
        print(f"Checkpoint loaded.")

    return epoch, all_train_loss, all_val_accu


def train(args, model, device, train_loader, optimizer, epoch):

    print("********************")
    print(f"***** EPOCH {epoch} ******")
    print("********************")
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(
            model=model,
            x_natural=data,
            y=target,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
        )
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

        # print progress
        if batch_idx % (len(train_loader) // 3) == 0:
            print(
                "[BATCH ({}) ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx + 1,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    train_loss = train_loss / len(train_loader)
    return train_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


import datetime
from time import sleep, time

from pgd_attack_data import eval_adv_test_whitebox


def main():

    print("Model will be saved at: ", args.model_dir)
    print("Number of classes: ", args.num_class)

    if not os.path.exists("%s" % (args.model_dir)):
        os.makedirs("%s" % (args.model_dir))
    Best_ADV_MODEL_FILE = str("%s/model_adv_best%03d.pth" % (args.model_dir, 0))

    ALL_TRAIN_LOSS = []
    ALL_VAL_ACCU = []
    best_val_adv_accu = 0.0

    model = WideResNet(num_classes=args.num_class).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        t0_epch = time()
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train_loss = train(args, model, device, train_loader, optimizer, epoch)

        print("Learning rate used: ", optimizer.param_groups[0]["lr"])

        print("================================================================")
        accuracy_clean, accuracy_robust = eval_adv_test_whitebox(
            model, device, test_loader
        )
        print("================================================================")

        ALL_TRAIN_LOSS.append(train_loss)
        ALL_VAL_ACCU.append([accuracy_clean, accuracy_robust])

        if accuracy_robust >= best_val_adv_accu:
            print("*** Best Robustness !")
            best_val_adv_accu = accuracy_robust
            torch.save(model, Best_ADV_MODEL_FILE)

        print(
            "\nTest set: Robust Accuracy: {:.2f}%    (Best: {:.2f}%)\n".format(
                accuracy_robust, best_val_adv_accu
            )
        )

        # Save the actual training state
        save_checkpoint(
            epoch,
            model,
            optimizer,
            ALL_TRAIN_LOSS,
            ALL_VAL_ACCU,
            args,
        )
        T_epch = time() - t0_epch
        # Print computation time
        print("Time: {} minutes".format(T_epch / 60))
        print(datetime.datetime.now())


if __name__ == "__main__":
    set_seed(args.seed)
    print(datetime.datetime.now())
    t0_epch_f = time()

    main()

    T_epch_f = time() - t0_epch_f
    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())
