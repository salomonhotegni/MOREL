"""
Train a model with a MOREL variant 
        (MOREL<--TRADES, MOREL<--MART or MOREL<--LOAT)
        on CIFAR10/100 or Tiny ImageNet.
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.vit import ViTForClassfication, get_config_vit, WrapperModel
from models.wideresnet import *  # input shape is (3, 32, 32) for CIFAR10/100
from models.resnet import *
from models.resnet_ti import * # input shape is (3, 64, 64) for Tiny ImageNet
from models.main_nets import main_net_cl

from misc_utils import *

import datetime
from time import time, sleep
from tqdm import tqdm

from datasets import load_dataset # pip install datasets

from eval_pgd_attack import eval_adv_test_whitebox


def train(args, model, device, train_loader, optimizer, epoch, init_epoch=1):
    if epoch >= init_epoch:
        print("************************")
        print(f"******* EPOCH {epoch} / {args.epochs} ********")
        print("************************")
    train_loss = 0.0
    orig_train_losses = torch.zeros([2, 1])
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if epoch < init_epoch:
            continue
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate morel loss
        loss, orig_loss, init_loss =  get_defense_loss(model,
                                                       data,
                                                       target,
                                                       optimizer,
                                                       epoch,
                                                       args)
        
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


def main(model, args, train_loader, test_loader, resume_train=[False, False]):

    if not os.path.exists("%s" % (args.model_dir)):
        os.makedirs("%s" % (args.model_dir))

    print("Training model with MOREL...")
    print("Architecture: ", args.arch)
    print("Dataset: ", args.data_name)
    print("Model will be saved at: ", args.model_dir)
    print(f"--- Preference vector: {args.k} ---")
    print(f"Loss function for accuracy: {args.accu_obj}")

    ALL_TRAIN_LOSS = []
    ALL_VAL_ACCU = []
    ALL_ORIG_LOSS = []
    last_best_val_accu_epch = None
    best_val_accu = 0.0
    best_val_adv_accu = 0.0
    best_val_clean_accu = 0.0

    model = model.to(args.device)
    if "vit" in args.arch:
        optimizer = optim.AdamW(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                milestones=args.optim_milestones, 
                                                gamma=args.optim_gamma)
    elif "resnet" in args.arch:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.optim_milestones, gamma=args.optim_gamma
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    
    init_epoch = 1
    if resume_train[0]:
        (
            last_epoch,
            ALL_TRAIN_LOSS,
            ALL_VAL_ACCU,
            ALL_ORIG_LOSS,
            last_best_val_accu_epch,
        ) = load_checkpoint(model, optimizer, scheduler, args.model_dir, loading_best=resume_train[1])
        if last_best_val_accu_epch is not None:
            best_val_accu, best_val_adv_accu, best_val_clean_accu, best_epch = (
                last_best_val_accu_epch
            )
            print("================================================================")
            print(f"Best Test Accuracy so far: {best_val_accu} at epoch {best_epch}.")
            print(f"Best Adv Accuracy so far: {best_val_adv_accu}..")
            print(f"Best Clean Accuracy so far: {best_val_clean_accu}.")
            print("================================================================")
        init_epoch = last_epoch + 1
        print(f"*** Resuming training from epoch {last_epoch + 1} ***")
    
    pbar = tqdm(total=init_epoch - 1, desc="Processing", unit="iteration")
    for epoch in range(1, args.epochs + 1):
        if epoch == init_epoch:
            pbar.close()  # *
        if epoch < init_epoch:
            sleep(0.1)  # *
            pbar.set_postfix(iteration=epoch)  # * -1
            pbar.update(1)

        t0_epch = time()
        scheduler.step()  # Update learning rate scheduler

        # adversarial training
        train_loss, orig_train_losses = train(
            args, model, device, train_loader, optimizer, epoch, init_epoch
        )
        if epoch < init_epoch:
            continue

        print("Learning rate used: ", optimizer.param_groups[0]["lr"])

        print("================================================================")
        accuracy_clean, accuracy_robust = eval_adv_test_whitebox(
            model, device, test_loader
        )
        print("================================================================")

        ALL_TRAIN_LOSS.append(train_loss)
        ALL_VAL_ACCU.append([accuracy_robust, accuracy_clean])
        ALL_ORIG_LOSS.append(orig_train_losses.numpy())
        
        gen_val_accuracy = (accuracy_robust + accuracy_clean) / 2
        
        if accuracy_clean >= best_val_clean_accu:
            print("*** Best Clean Accuracy !")
            best_val_clean_accu = accuracy_clean

        if gen_val_accuracy >= best_val_accu:
            print("---- Best Overalll Performance !")
            best_val_accu = gen_val_accuracy

        if accuracy_robust >= best_val_adv_accu:
            print("*** Best Robust Accuracy !")
            best_val_adv_accu = accuracy_robust
            last_best_val_accu_epch = [
                best_val_accu,
                best_val_adv_accu,
                best_val_clean_accu,
                epoch,
            ]
            # This is denoted as 'best' in the paper
            save_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                ALL_TRAIN_LOSS,
                ALL_VAL_ACCU,
                args,
                ALL_ORIG_LOSS=ALL_ORIG_LOSS,
                best_val_accuracy_epch=last_best_val_accu_epch,
                saving_best=True,
                verbose=False,
            )

        print(
            "\nTest set: Overall Accuracy: {:.2f}%    (Best: {:.2f}%)\n".format(
                gen_val_accuracy, best_val_accu
            )
        )

        # Save the actual training state
        save_checkpoint(
            epoch,
            model,
            optimizer,
            scheduler,
            ALL_TRAIN_LOSS,
            ALL_VAL_ACCU,
            args,
            ALL_ORIG_LOSS = ALL_ORIG_LOSS,
            best_val_accuracy_epch=last_best_val_accu_epch,
            saving_best=False,
        )
        T_epch = time() - t0_epch
        # Print computation time
        print("Time: {} minutes".format(T_epch / 60))
        print(datetime.datetime.now())

if __name__ == "__main__":
    #### General settings for MOREL
    parser = argparse.ArgumentParser(description="MOREL Adversarial Training")
    # Method
    parser.add_argument(
            "--method",
            default="morel",
            choices=["morel"],
        )
    parser.add_argument(
            "--accu-obj",
            default="mart",
            help="Loss function for accuracy ('trades', 'mart' or 'loat')",
        )
    parser.add_argument("--k", default=[0.1, 0.9])
    parser.add_argument("--a", default=[0] * 2, type=int)
    parser.add_argument("--gamma", default=2e-5, type=float)
    parser.add_argument(
            "--alpha",
            default=1e-5,
            type=float,
            help="Coefficient for Contrastive Loss",
        )

    parser.add_argument("--beta", default=6.0, help="weight before kl")
    # Training and evaluation
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
            "--no-cuda", action="store_true", default=False, help="disables CUDA training"
        )
    parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
    # Attack settings (PGD)
    parser.add_argument("--epsilon", default=0.031, help="perturbation")
    parser.add_argument("--num-steps", default=10, help="perturb number of steps for training")
    parser.add_argument("--step-size", default=0.007, help="perturb step size for training")
        
    # Data
    parser.add_argument("--data_name", default="cifar10")
    # Model
    parser.add_argument("--arch", 
                            default="vit_small",
                            choices=["resnet_18", "wide_resnet_34_10", "vit_small"],
                            help="model architecture to use for training"
                            ) 
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--num_att_heads", default=2, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
            
    args =  parser.parse_args()

    args.model_dir = "logs/morel/"+"with_"+args.accu_obj+"/"\
        + args.data_name + "/model-" + args.data_name\
            + "-" + args.arch + "-v01"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}
    args.device = device

    print("Using device: ", args.device)

    #### Architecture specific settings
    if "vit" in args.arch:
        args.epochs = 2 #50 # number of epochs for training ViT
        # AdamW optimizer settings
        args.weight_decay = 0.05
        args.lr = 0.0001
        args.optim_milestones = [25]
        args.optim_gamma = 1/10
    elif "resnet" in args.arch:
        args.epochs = 2 #100 # number of epochs for training ResNet
        # SGD optimizer settings
        args.momentum = 0.9
        args.weight_decay = 1e-4
        args.optim_milestones = [75, 90]
        
        if "wide" in args.arch:
            args.lr = 0.01
            args.optim_gamma = 1/100
        else:
            args.lr = 0.001
            args.optim_gamma = 1/10
            
    #### Method specific settings
    if "loat" in args.accu_obj:
        args.beta = 6.0  # weight before kl (misclassified examples)
        args.LORE_type = "LORE_v1"  # LORE type
        args.reg = True  # use regularization
        args.reg_type = "mse"  # regularization type
        args.from_epoch = 1 
        args.adv = True  
        args.gamma_loat = 0.05  
        args.theta = 1.0  

    #### Setup data loaders
    if "cifar" in args.data_name:
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
    elif "imagenet" in args.data_name:
        transform_train = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),  # Convert to RGB if not already
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),  # Convert to RGB if not already
                transforms.ToTensor(),
            ]
        )
        
    print("###### DATASET: ", args.data_name)
    if args.data_name == "tiny_imagenet":
        args.num_class = 200
        dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir="data/tiny_imagenet")
        # Create PyTorch-compatible datasets
        trainset = TinyImageNetDataset(dataset["train"], transform=transform_train)
        testset = TinyImageNetDataset(dataset["valid"], transform=transform_test)
    elif args.data_name == "cifar100":
        args.num_class = 100
        trainset = torchvision.datasets.CIFAR100(
            root="data/cifar100", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="data/cifar100", train=False, download=True, transform=transform_test
        )
    elif args.data_name == "cifar10":
        args.num_class = 10
        trainset = torchvision.datasets.CIFAR10(
            root="data/cifar10", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="data/cifar10", train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    
    ######################### START TRAINING #########################
    
    set_seed(args.seed)
    print(datetime.datetime.now())
    t0_epch_f = time()
    
    if args.arch == "wide_resnet_34_10":
        if args.data_name == "tiny_imagenet":
            raise ValueError("The available WideResNet is not supported for Tiny ImageNet dataset.")
        base_model = WideResNet(num_classes=args.num_class)
    elif args.arch == "resnet_18":
        if args.data_name == "tiny_imagenet":
            base_model = ResNet18_tiny(num_classes=args.num_class)
        else:
            base_model = ResNet18(num_classes=args.num_class)
    elif "vit" in args.arch:
        if args.data_name == "tiny_imagenet":
            raise ValueError("The available ViT is not supported for Tiny ImageNet dataset.")
        config_vit=get_config_vit(
                num_class=args.num_class,
                hidden_size = 256,  # Default hidden size for ViT
                num_hidden_layers = 1,  # Default number of hidden layers
            )
        base_model = ViTForClassfication(config_vit)
        base_model = WrapperModel(base_model)  # Wrap the ViT model
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    
    if "vit" in args.arch:
        mod_encoder = base_model
        args.feat_dim = base_model.classifier.in_features
        mod_encoder.classifier = torch.nn.Identity() # Remove classifier from Encoder

    elif "resnet" in args.arch:
        mod_encoder = base_model
        args.feat_dim = mod_encoder.fc.in_features
        mod_encoder.fc = torch.nn.Identity()  # Remove classifier from Encoder
    
    head_classifier = nn.Linear(args.feat_dim, args.num_class)
    model = main_net_cl(mod_encoder, head_classifier, args)

    # Print the number of learnable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")

    main(model, 
         args,
         train_loader, test_loader,
         resume_train=[False, False],  # (Resume Training ?, From the best epoch ?)
         )

    T_epch_f = time() - t0_epch_f
    # Print computation time
    convert_seconds(T_epch_f)
    print(datetime.datetime.now())