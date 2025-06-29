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


def _pgd_whitebox(
    model,
    X,
    y,
    device,
    epsilon=0.031, # 8.0 / 255.0
    num_steps=20,
    step_size=0.003, # (8.0 / 255.0) / 10.0
    bounds=(0.0, 1.0),
    get_X_adv=False,
    use_random=True
):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if use_random:
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
    device,
    epsilon=0.031,  # 8.0 / 255.0
    num_steps=20,
    step_size=0.003,  # (8.0 / 255.0) / 10.0
    bounds=(0.0, 1.0),
    use_random=True
):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if use_random:
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
        err_natural, err_robust = _pgd_whitebox(model, X, y, device=device)
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
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y, device=device)
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