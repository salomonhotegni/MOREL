import os
import torch
import torch.nn as nn
import numpy as np
import random

from losses.loat import loat_m_loss
from losses.mart import mart_loss
from losses.trades import trades_loss
from losses.morel import morel_loss

from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    """for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_checkpoint(
    model,
    model_dir,
    filename,
    verbose=True,
):

    filename = str(f"{model_dir}/{filename}.pth")
    if torch.cuda.is_available():
        checkpoint = torch.load(filename, weights_only=False)
    else:
        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if verbose:
        print("Model loaded from: ", filename)

    return model


def convert_seconds(seconds):
    # Calculate hours, minutes, and seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    # Format the result
    print(f"Time: {hours} hours {minutes} minutes {remaining_seconds} seconds")
    
def get_defense_loss(model, X, y, optimizer, epoch, args):
    #### MOREL
    if "morel" in args.method:
        if "trades" in args.accu_obj or "mart" in args.accu_obj:
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
            extra_configs = None,
            extra_outputs=True,
        )
        if "loat" in args.accu_obj:
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

    #### SOO
    if args.method == "mart":
        return mart_loss(
                model=model,
                x_natural=X,
                y=y,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
            )
    elif args.method == "trades":
        return trades_loss(
                model=model,
                x_natural=X,
                y=y,
                optimizer=optimizer,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.num_steps,
                beta=args.beta,
            )
    elif args.method == "loat":
        return loat_m_loss(model=model,
                           epoch=epoch,
                           x_natural=X,
                           y=y,
                           reg=args.reg,
                           reg_type=args.reg_type,
                           optimizer=optimizer,
                           device = args.device,
                           classes = args.num_class,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           theta=args.theta,
                           gamma=args.gamma_loat,
                           lot=args.LORE_type)
    else:
        raise ValueError(f"Unknown loss function: {args.method}")

class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get the image and label from Hugging Face dataset
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]

        # Apply the transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label
    

def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    ALL_TRAIN_LOSS,
    ALL_VAL_ACCU,
    args,
    ALL_ORIG_LOSS=None,
    best_val_accuracy_epch=None,
    saving_best=False,
    verbose=True,
):
    model_dir = args.model_dir
    if not os.path.exists("%s" % (model_dir)):
        os.makedirs("%s" % (model_dir))
    TRAIN_STATE_FILE = str("%s/last_train_checkpoint.pth" % (model_dir))
    BESTTRAIN_STATE_FILE = str("%s/best_train_checkpoint.pth" % (model_dir))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        "last_best_valid_accuracy_epoch": best_val_accuracy_epch,
        "list_train_loss": ALL_TRAIN_LOSS,
        "list_val_accuracy": ALL_VAL_ACCU,
        "list_orig_loss": ALL_ORIG_LOSS,
        "args": args,
    }
    if saving_best:
        torch.save(checkpoint, BESTTRAIN_STATE_FILE)
    else:
        torch.save(checkpoint, TRAIN_STATE_FILE)

    if verbose:
        print(f"Checkpoint saved for epoch {epoch}.")

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    model_dir,
    loading_best=False,
    verbose=True,
):
    if loading_best:
        filename = str("%s/best_train_checkpoint.pth" % (model_dir))
    else:
        filename = str("%s/last_train_checkpoint.pth" % (model_dir))

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"]
    last_best_val_accu_epch = checkpoint["last_best_valid_accuracy_epoch"]
    all_train_loss = checkpoint["list_train_loss"]
    all_val_accu = checkpoint["list_val_accuracy"]
    all_orig_loss = checkpoint["list_orig_loss"]

    if verbose:
        print(f"Checkpoint loaded.")

    return (
        epoch,
        all_train_loss,
        all_val_accu,
        all_orig_loss,
        last_best_val_accu_epch,
    )