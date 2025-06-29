"""
Test a trained model on CIFAR10/100 or Tiny ImageNet.
"""

from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *  # input shape is (3, 32, 32) for CIFAR10/100
from models.resnet import *
from models.wideresnet_ti import * # input shape is (3, 64, 64) for Tiny ImageNet
from models.resnet_ti import *
from misc_utils import *
from models.main_nets import main_net_cl

from models.vit import ViTForClassfication, get_config_vit, WrapperModel

import datetime
from time import time, sleep

from datasets import load_dataset # pip install datasets

from test_attacks_utils import testing_adv_whitebox, testing_adv_blackbox

import torch
from torchvision import models
from torchvision import transforms

from collections import OrderedDict
from typing import Tuple
from torch import Tensor
import torch

parser = argparse.ArgumentParser(description="PyTorch Attack Evaluation")
# Method
parser.add_argument(
    "--method",
    default="morel_m",
    help="trades, morel_t, mart, morel_m, loat or morel_l",
)
# Testing
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
    choices=["best", "last"],
    help="Type of the model: 'best' or 'last' ?",
)

# PGD / FGSM
parser.add_argument("--epsilon", default=0.031, help="perturbation")
parser.add_argument("--num-steps", default=20, help="perturb number of steps")
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

# Data
parser.add_argument("--data_name", default="cifar10")
# Model
parser.add_argument("--arch", 
                        default="vit_small",
                        choices=["resnet_18", "wide_resnet_34_10", "vit_small"],
                        help="model architecture to use for training"
                        ) 
parser.add_argument("--embed_dim", default=128, type=int) # embedding dimension
parser.add_argument("--num_att_heads", default=2, type=int) # number of attention heads
parser.add_argument("--dropout", default=0.0, type=float) # dropout rate

args = parser.parse_args()


class Identity(torch.nn.Module):
    def __init__(self, activation=False):
        super(Identity, self).__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation=="sigmoid":
            #[0,1]
            return torch.sigmoid(x)*2-1 #(1 + torch.sigmoid(x)) / 2
            #[-1,1]
            #return torch.sigmoid(x)*4-3
        elif self.activation=="relu":
            return torch.relu(x)
        else:
            return x

class ImageNormalizer(torch.nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std
    
def normalize_model(model: torch.nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float], device:str) -> torch.nn.Module:

    if "cuda" in device:
        device = "cuda:0"

    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return torch.nn.Sequential(layers).to(device)


if "morel" in args.method:
    if "morel_t" in args.method:
        args.accu_obj = "trades"
    elif "morel_m" in args.method:
        args.accu_obj = "mart"
    elif "morel_l" in args.method:
        args.accu_obj = "loat"
    else:
        raise ValueError("Unknown method: %s" % args.method)
    args.model_dir = "logs/morel/"+"with_"+args.accu_obj+"/"\
        + args.data_name + "/model-" + args.data_name\
            + "-" + args.arch + "-v01"
else:
    args.model_dir = "logs/"+args.method+"/"\
    + args.data_name + "/model-" + args.data_name\
        + "-" + args.arch + "-v01"

args.model_source_dir = "logs/standard/"\
    + args.data_name + "/model-" + args.data_name + "-resnet50-v01"

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

args.device = device

if __name__ == "__main__":

    if "cifar" in args.data_name:
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        args.input_shape = (3, 32, 32)  # CIFAR input shape
    elif "imagenet" in args.data_name:
        transform_test = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),  # Convert to RGB if not already
                transforms.ToTensor(),
            ]
        )
        args.input_shape = (3, 64, 64)  # Tiny-ImageNet input shape
        
    print("###### DATASET: ", args.data_name)
    if args.data_name == "tiny_imagenet":
        args.num_class = 200
        dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir="data/tiny_imagenet")
        testset = TinyImageNetDataset(dataset["valid"], transform=transform_test)
    elif args.data_name == "cifar100":
        args.num_class = 100
        testset = torchvision.datasets.CIFAR100(
            root="data/cifar100", train=False, download=True, transform=transform_test
        )
    elif args.data_name == "cifar10":
        args.num_class = 10
        testset = torchvision.datasets.CIFAR10(
            root="data/cifar10", train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    
    for which_model in ["best", "last"]:
        args.which_model = which_model

        if "morel" in args.method:
            if args.arch == "wide_resnet_34_10":
                if args.data_name == "tiny_imagenet":
                    raise ValueError("The available WideResNet is not supported for Tiny ImageNet dataset.")
                base_model = WideResNet(num_classes=args.num_class)
                args.feat_dim = base_model.fc.in_features
                base_model.fc = torch.nn.Identity()  # Remove classifier from Encoder
            elif args.arch == "resnet_18":
                if args.data_name == "tiny_imagenet":
                    base_model = ResNet18_tiny(num_classes=args.num_class)
                else:
                    base_model = ResNet18(num_classes=args.num_class)
                args.feat_dim = base_model.fc.in_features
                base_model.fc = torch.nn.Identity()  # Remove classifier from Encoder
            elif "vit" in args.arch:
                if args.data_name == "tiny_imagenet":
                    raise ValueError("The available ViT is not supported for Tiny ImageNet dataset.")
                config_vit=get_config_vit(
                    num_class=args.num_class,
                    hidden_size = 256,  
                    num_hidden_layers = 1,  
                )
                base_model = ViTForClassfication(config_vit)
                base_model = WrapperModel(base_model)
                args.feat_dim = base_model.classifier.in_features
                base_model.classifier = torch.nn.Identity()  # Remove classifier from Encoder
                
            mod_encoder = base_model
            head_classifier = nn.Linear(args.feat_dim, args.num_class)
            model = main_net_cl(mod_encoder, head_classifier, args)
        else:
            if args.arch == "wide_resnet_34_10":
                model = WideResNet(num_classes=args.num_class)
            elif args.arch == "resnet_18":
                model = ResNet18(num_classes=args.num_class)
            elif "vit" in args.arch:
                config_vit=get_config_vit(
                            num_class=args.num_class,
                            hidden_size = 256,
                            num_hidden_layers = 1,
                        )
                model = ViTForClassfication(config_vit)
                model = WrapperModel(model)
            else:
                raise ValueError("Unknown architecture: %s" % args.arch)

        model = model.to(args.device)

        if "best" in args.which_model:
            model = load_model_checkpoint(
                    model, args.model_dir, filename="best_train_checkpoint",
                    verbose=True
                )
            print("------ BEST MODEL CONSIDERED ------")
        else:
            model = load_model_checkpoint(
                    model, args.model_dir, filename="last_train_checkpoint",
                    verbose=True
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
            ["auto", 0],
        ]:
  
            args.attack, args.num_steps = at[0], at[1]
            print(datetime.datetime.now())
            t0_epch_f = time()
            if args.is_whiteBox:
                print(f"Testing with: {args.attack} white-box attack")
                if "auto" not in args.attack and "cw" not in args.attack:
                    print(f"Num steps: {args.num_steps}")
                accuracy_clean, accuracy_robust = testing_adv_whitebox(
                        model, device, test_loader, args, resume_test=False
                    )
            else:
                print(f"Testing with: {args.attack} black-box attack")
                if "auto" not in args.attack and "cw" not in args.attack:
                    print(f"Num steps: {args.num_steps}")
                accuracy_clean, accuracy_robust = testing_adv_blackbox(
                        model, model_adv, device, test_loader, args, resume_test=False
                    )
            T_epch_f = time() - t0_epch_f
            # Print computation time
            convert_seconds(T_epch_f)
            print(datetime.datetime.now())