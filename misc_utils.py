import torch
import torch.nn as nn
import numpy as np
import random


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
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
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location="cpu")
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
