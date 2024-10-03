from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from models.main_nets import main_net_cl
from torchvision.models import resnet18

from morel import morel_loss

from misc_utils import *

import datetime
from time import time, sleep

from pgd_attack_data import eval_adv_test_whitebox

parser = argparse.ArgumentParser(description="PyTorch CIFAR MOREL Adversarial Training")
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    metavar="N",
    help="input batch size for training",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=200,
    metavar="N",
    help="input batch size for testing",
)

parser.add_argument(
    "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
)
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W")
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate"
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument("--epsilon", default=0.031, help="perturbation")
parser.add_argument("--num-steps", default=10, help="perturb number of steps")
parser.add_argument("--step-size", default=0.007, help="perturb step size")
parser.add_argument("--beta", default=6.0, help="weight before kl (in TRADES and MART)")
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--model-dir",
    default="logs/morel/with_mart/cifar10/final_wrsnt34_10_morel_0.1_0.9_v01",
    help="directory of model for saving checkpoint",
)
parser.add_argument(
    "--accu-obj",
    default="mart",
    help="Loss function for accuracy ('trades' or 'mart')",
)

# ConeScalarization
parser.add_argument("--k", default=[0.1, 0.9])
parser.add_argument("--a", default=[0] * 2, type=int)
parser.add_argument("--gamma", default=2e-5, type=float)

# Model
parser.add_argument("--arch", default="wide_resnet_34_10")
parser.add_argument(
    "--feat_dim", default=640
)  # Encoder output dimension: depends on the architecture of the original model
parser.add_argument("--embed_dim", default=128, type=int)
parser.add_argument("--num_att_heads", default=2, type=int)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--num_class", default=10, type=int)

parser.add_argument(
    "--alpha",
    default=1e-5,
    type=float,
    help="Coefficient for Contrastive Loss",
)
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

args.device = device

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
    ALL_ORIG_LOSS,
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
        "list_original_loss": ALL_ORIG_LOSS,
        "args": args,
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
    list_original_loss = checkpoint["list_original_loss"]

    if verbose:
        print(f"Checkpoint loaded.")

    return (
        epoch,
        all_train_loss,
        all_val_accu,
        list_original_loss,
    )


def train(args, model, device, train_loader, optimizer, epoch):
    print("********************")
    print(f"***** EPOCH {epoch} ******")
    print("********************")
    train_loss = 0.0
    orig_train_losses = torch.zeros([2, 1])
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, orig_loss, init_loss = morel_loss(
            model=model,
            x_natural=data,
            y=target,
            optimizer=optimizer,
            k=args.k,
            a=args.a,
            gamma=args.gamma,
            accu_obj=args.accu_obj,
            alpha=args.alpha,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            extra_outputs=True,
        )

        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        orig_train_losses += orig_loss

        # print progress
        if batch_idx % (len(train_loader) // 3) == 0:
            print(
                "[BATCH ({}) ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx + 1,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            print(
                "**Losses** | CE: {:.6f} | KL: {:.6f} | COS: {:.6f} | CS: {:.6f} |".format(
                    init_loss[0],
                    init_loss[1],
                    init_loss[2],
                    init_loss[3],
                )
            )

    train_loss = train_loss / len(train_loader)
    orig_train_losses = orig_train_losses / len(train_loader)
    return train_loss, orig_train_losses


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr

    if epoch >= 90:
        lr = args.lr / 100**2
    elif epoch >= 75:
        lr = args.lr / 100

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main(model, args):

    if not os.path.exists("%s" % (args.model_dir)):
        os.makedirs("%s" % (args.model_dir))
    Best_ADV_MODEL_FILE = str("%s/model_adv_best%03d.pth" % (args.model_dir, 0))

    print("Model will be saved at: ", args.model_dir)
    print(f"--- Preference vector: {args.k} ---")
    print(f"Loss function for accuracy: {args.accu_obj}")
    print("Number of classes: ", args.num_class)

    ALL_TRAIN_LOSS = []
    ALL_VAL_ACCU = []
    ALL_ORIG_LOSS = []
    best_val_adv_accu = 0.0

    model = model.to(args.device)
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
        train_loss, orig_train_losses = train(
            args, model, device, train_loader, optimizer, epoch
        )

        print("Learning rate used: ", optimizer.param_groups[0]["lr"])

        print("================================================================")
        accuracy_clean, accuracy_robust = eval_adv_test_whitebox(
            model, device, test_loader
        )
        print("================================================================")

        ALL_TRAIN_LOSS.append(train_loss)
        ALL_VAL_ACCU.append([accuracy_robust, accuracy_clean])
        ALL_ORIG_LOSS.append(orig_train_losses.numpy())

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
            ALL_ORIG_LOSS,
            args,
        )
        T_epch = time() - t0_epch
        # Print computation time
        print(f"Time: {T_epch / 60} minutes")
        print(datetime.datetime.now())


if __name__ == "__main__":
    set_seed(args.seed)
    print(datetime.datetime.now())
    t0_epch_f = time()

    if args.arch == "wide_resnet_34_10":
        base_model = WideResNet(num_classes=args.num_class)
    elif args.arch == "resnet_18":
        base_model = ResNet18(num_classes=args.num_class)

    mod_encoder = base_model
    mod_encoder.fc = torch.nn.Identity()  # Remove classifier from Encoder
    head_classifier = nn.Linear(args.feat_dim, args.num_class)
    model = main_net_cl(mod_encoder, head_classifier, args)

    # Print the number of learnable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")

    main(model, args)

    T_epch_f = time() - t0_epch_f
    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())
