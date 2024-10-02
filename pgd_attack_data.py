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
    "--epsilon",
    default=0.031,
    # 8.0 / 255.0,
    help="perturbation",
)
parser.add_argument("--num-steps", default=20, help="perturb number of steps")
parser.add_argument(
    "--step-size",
    default=0.003,
    # (8.0 / 255.0) / 10.0,
    help="perturb step size",
)
parser.add_argument("--random", default=True, help="random initialization for PGD")
parser.add_argument(
    "--white-box-attack", default=True, help="whether perform white-box attack"
)

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

args.device = device

# set up data loader
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def _pgd_whitebox(
    model,
    X,
    y,
    epsilon=args.epsilon,
    num_steps=args.num_steps,
    step_size=args.step_size,
    bounds=(0.0, 1.0),
    get_X_adv=False,
):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, bounds[0], bounds[1]), requires_grad=True)

    if get_X_adv:
        return X_pgd
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    return err, err_pgd


def _pgd_blackbox(
    model_target,
    model_source,
    X,
    y,
    epsilon=args.epsilon,
    num_steps=args.num_steps,
    step_size=args.step_size,
    bounds=(0.0, 1.0),
):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = (
            torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        )
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, bounds[0], bounds[1]), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()

    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    total_samples = len(test_loader.dataset)
    accuracy_clean = (1 - (natural_err_total / total_samples)) * 100
    accuracy_clean = accuracy_clean.detach().item()
    accuracy_robust = (1 - (robust_err_total / total_samples)) * 100
    accuracy_robust = accuracy_robust.detach().item()
    print("Clean Accuracy: ", accuracy_clean)
    print("Robust Accuracy: ", accuracy_robust)

    return [accuracy_clean, accuracy_robust]


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    total_samples = len(test_loader.dataset)
    accuracy_clean = (1 - (natural_err_total / total_samples)) * 100
    accuracy_clean = accuracy_clean.detach().item()
    accuracy_robust = (1 - (robust_err_total / total_samples)) * 100
    accuracy_robust = accuracy_robust.detach().item()
    print("Clean Accuracy: ", accuracy_clean)
    print("Robust Accuracy: ", accuracy_robust)

    return [accuracy_clean, accuracy_robust]


def main(test_loader):

    if args.white_box_attack:
        # white-box attack
        print("pgd white-box attack")
        model = WideResNet().to(device)
        model.load_state_dict(torch.load(args.model_path))

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print("pgd black-box attack")
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == "__main__":
    testset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs
    )
    main(test_loader)
