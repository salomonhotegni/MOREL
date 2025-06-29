from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
from time import time, sleep
from tqdm import tqdm

from advermorel.utils.misc import *
from advermorel.losses.morel import morel_loss

def get_morel_loss(model, X, y, 
                     optimizer, epoch, args, 
                     extra_configs=None,
                     custom_accu_obj=None):
    """
    Computes the MOREL loss for the given model and data.
    Args:
        model: The model to train.
        X: Input data.
        y: Target labels.
        optimizer: Optimizer for the model.
        epoch: Current epoch number.
        args: Arguments containing hyperparameters and configurations.
        extra_configs: Additional configurations for the loss function.
        custom_accu_obj: Custom accuracy loss function if provided.
                        It should accept the following parameters:
                        model, x_natural, y, optimizer, and extra_configs,
                        where extra_configs is a dictionary
                        containing any additional configurations needed by the custom loss function.
                        It should return a tuple of (loss for accuracy, adversarial batch).
    Returns:
        ConeScal_loss: The computed MOREL loss.
        orig_loss: Original losses (robust loss and clean loss).
        init_loss: Initial losses (if applicable).
    """
    if args.accu_obj is None:
        return morel_loss(
                model=model,
                x_natural=X,
                y=y,
                optimizer=optimizer,
                k=args.k,
                a=args.a,
                gamma=args.gamma,
                accu_obj=args.accu_obj,
                alpha=args.alpha,
                extra_configs=extra_configs,
                custom_accu_obj=custom_accu_obj,
            )
    elif "trades" in args.accu_obj or "mart" in args.accu_obj:
            return morel_loss(
            model=model,
            x_natural=X,
            y=y,
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
            extra_configs=None,
            extra_outputs=True,
        )
    elif "loat" in args.accu_obj:
            return morel_loss(
                model=model,
                x_natural=X,
                y=y,
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
                extra_configs={"epoch": epoch,
                            "reg": args.reg,
                            "reg_type": args.reg_type,
                            "device": args.device,
                            "theta": args.theta,
                            "gamma_loat": args.gamma_loat,
                            "lot": args.LORE_type,
                },
                extra_outputs=True,
            )
    else:
        raise ValueError(f"Loss function {args.accu_obj} not implemented !")

def train_morel_epoch(args, model, device, 
                train_loader, optimizer, epoch,
                custom_accu_obj=None,
                extra_configs=None,
                init_epoch=1):
    """
    Train the model for one epoch using the MOREL loss.
    Args:
        args: Arguments containing hyperparameters and configurations.
        model: The model to train.
        device: Device to run the training on (CPU or GPU).
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for the model.
        epoch: Current epoch number.
        custom_accu_obj: Custom accuracy loss function if provided.
                        It should accept the following parameters:
                        model, x_natural, y, optimizer, and extra_configs,
                        where extra_configs is a dictionary
                        containing any additional configurations needed by the custom loss function.
                        It should return a tuple of (loss for accuracy, adversarial batch).
        extra_configs: Additional configurations for the loss function.
        init_epoch: Initial epoch to start training from.
    Returns:
        train_loss: Average training loss for the epoch.
        orig_train_losses: Original losses (robust loss and clean loss) for the epoch.
    """
    
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
        loss, orig_loss, init_loss =  get_morel_loss(model, data, target,
                                                       optimizer, epoch, args,
                                                       extra_configs=extra_configs, 
                                                       custom_accu_obj=custom_accu_obj)

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
            if init_loss is not None:
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