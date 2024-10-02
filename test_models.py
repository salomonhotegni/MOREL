from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from misc_utils import *
from models.main_nets import main_net_cl

import datetime
from time import time, sleep

from test_attacks_utils import testing_adv_whitebox, testing_adv_blackbox

parser = argparse.ArgumentParser(description="PyTorch CIFAR PGD Attack Evaluation")
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=200,
    metavar="N",
    help="input batch size for testing (default: 200)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)

parser.add_argument(
    "--is-whiteBox",
    default=True,
    help="White-Box attack - or - Black-Box attack ?",
)
parser.add_argument(
    "--which-model",
    default="best",
    help="Type of the model: 'best' or 'last' ?",
)
parser.add_argument(
    "--attack",
    default="cw_inf",
    help="adversarial attack name ['pgd', 'fgsm', 'cw_inf']",
)
# PGD / FGSM
parser.add_argument("--epsilon", default=0.031, help="perturbation")
parser.add_argument("--num-steps", default=100, help="perturb number of steps")
parser.add_argument("--step-size", default=0.003, help="perturb step size")
parser.add_argument("--random", default=True, help="random initialization for PGD")
# CW-inf
parser.add_argument("--max-steps-cw", default=10, help="max iterations for CW-inf")
parser.add_argument(
    "--cw-lr",
    default=1e-2,
    help="The initial learning rate for CW-inf attack algorithm",
)
parser.add_argument(
    "--confidence", default=1, help="confidence of the adversarial examples"
)
parser.add_argument("--init_const_c", default=15, help="The initial value of constant")


parser.add_argument(
    "--model-dir",
    default="logs/morel/with_mart/cifar10/final_wrsnt34_10_morel_0.1_0.9_v01",
    help="directory of the model",
)
parser.add_argument(
    "--model-source-dir",
    default="logs/standard/cifar10/model-cifar10-resnet50-v01",
    help="directory of source model for black-box attack",
)

parser.add_argument(
    "--method",
    default="morel_m",
    help="trades, mart, morel_t or morel_m",
)

# Model
parser.add_argument("--arch", default="wide_resnet_34_10")
parser.add_argument(
    "--feat_dim", default=640
)  # Encoder output dimension: depends on the architecture of the original model
parser.add_argument("--embed_dim", default=128, type=int)
parser.add_argument("--num_att_heads", default=2, type=int)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--num_class", default=10, type=int)

parser.add_argument("--input_shape", default=(3, 32, 32))

# Data
parser.add_argument("--data_name", default="cifar10")

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

args.device = device


if __name__ == "__main__":

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if args.data_name == "cifar100":
        testset = torchvision.datasets.CIFAR100(
            root="data/cifar100", train=False, download=True, transform=transform_test
        )
        print("###### DATASET: ", args.data_name)
    else:
        testset = torchvision.datasets.CIFAR10(
            root="data/cifar10", train=False, download=True, transform=transform_test
        )
        print("###### DATASET: ", args.data_name)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )

    for which_model in ["best", "last"]:
        args.which_model = which_model

        if "morel" in args.method:
            if args.arch == "wide_resnet_34_10":
                base_model = WideResNet(num_classes=args.num_class)
            elif args.arch == "resnet_18":
                base_model = ResNet18(num_classes=args.num_class)
            mod_encoder = base_model
            mod_encoder.fc = torch.nn.Identity()  # Remove classifier from Encoder
            head_classifier = nn.Linear(args.feat_dim, args.num_class)
            model = main_net_cl(mod_encoder, head_classifier, args)
        else:
            if args.arch == "wide_resnet_34_10":
                model = WideResNet(num_classes=args.num_class)
            elif args.arch == "resnet_18":
                model = ResNet18(num_classes=args.num_class)

        model = model.to(args.device)

        if "best" in args.which_model:
            MODEL_FILE = str("%s/model_adv_best%03d.pth" % (args.model_dir, 0))
            if torch.cuda.is_available():
                model = torch.load(MODEL_FILE)
            else:
                model = torch.load(MODEL_FILE, map_location="cpu")
            print("Model loaded from: ", MODEL_FILE)
            print("------ BEST MODEL CONSIDERED ------")
        else:
            model = load_model_checkpoint(
                model, args.model_dir, filename="last_train_checkpoint"
            )
            print("------ LAST MODEL CONSIDERED ------")

        # Source model for black-box attack
        if not args.is_whiteBox:
            model_adv = ResNet50(num_classes=args.num_class).to(args.device)
            model_adv = load_model_checkpoint(
                model_adv, args.model_source_dir, filename="last_train_checkpoint"
            )

        print("Num Batches in Test Loader: ", len(test_loader))

        for at in [
            ["fgsm", 1],
            ["pgd", 20],
            ["pgd", 100],
            ["cw_inf", args.max_steps_cw],
        ]:
            args.attack, args.num_steps = at[0], at[1]
            print(datetime.datetime.now())
            t0_epch_f = time()
            if args.is_whiteBox:
                print(f"Testing with: {args.attack} white-box attack")
                print(f"Num steps: {args.num_steps}")
                accuracy_clean, accuracy_robust = testing_adv_whitebox(
                    model, device, test_loader, args
                )
            else:
                print(f"Testing with: {args.attack} black-box attack")
                print(f"Num steps: {args.num_steps}")
                accuracy_clean, accuracy_robust = testing_adv_blackbox(
                    model, model_adv, device, test_loader, args
                )
            T_epch_f = time() - t0_epch_f
            # Print computation time
            convert_seconds(T_epch_f)
            print(datetime.datetime.now())
