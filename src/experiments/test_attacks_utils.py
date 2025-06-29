import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datetime
from time import time, sleep

from misc_utils import *

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    ProjectedGradientDescent,
    FastGradientMethod,
    CarliniLInfMethod,
    AutoAttack,
)

def get_attack_art(model, optimizer, args, clip_values=(0.0, 1.0)):
    classifier_att = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=args.input_shape,
        nb_classes=args.num_class,
    )

    if args.attack == "pgd":
        attack = ProjectedGradientDescent(
            estimator=classifier_att,
            norm=np.inf,
            eps=args.epsilon,
            eps_step=args.step_size,
            max_iter=args.num_steps,
            targeted=False,
            num_random_init=0,
            batch_size=args.test_batch_size,
        )

    elif args.attack == "fgsm":
        attack = FastGradientMethod(
            estimator=classifier_att,
            norm=np.inf,
            eps=args.epsilon,
            eps_step=args.step_size,
            targeted=False,
            num_random_init=0,
            batch_size=args.test_batch_size,
        )

    elif args.attack == "cw_inf":
        attack = CarliniLInfMethod(
            classifier=classifier_att,
            targeted=False,
            initial_const=args.init_const_c,
            learning_rate=args.cw_lr,
            max_iter=args.max_steps_cw,
            batch_size=args.test_batch_size,
        )

    elif args.attack == "auto":
        attack = AutoAttack(
            estimator=classifier_att,
            norm=np.inf,
            eps=args.epsilon,
            eps_step=args.step_size,
            batch_size=args.test_batch_size,
            estimator_orig=None,
            targeted=False,
        )

    else:
        err_str = f"Attack {args.attack} is not supported"
        raise AssertionError(err_str)

    return attack


def gen_adv_data(data, 
                 targets, 
                 model, 
                 optimizer, 
                 args, 
                 bounds=(0.0, 1.0)):
    attack = get_attack_art(model, optimizer, args, bounds)
    data_attack = attack.generate(data.cpu().numpy(), y=targets.cpu().numpy())
    data_attack = torch.from_numpy(data_attack).to(args.device)
    return data_attack


def testing_adv_whitebox(model, device, test_loader, args, resume_test=False):
    """
    evaluate model by white-box attack
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
    )
    model.eval()
    correct_clean = 0
    correct_adv = 0
    current_batch = -1

    if resume_test:
        checkpoint_test = torch.load(
            str(
                "%s/%s_current_whitetest_checkpoint.pth" % (args.model_dir, args.attack)
            )
        )
        current_batch = checkpoint_test["batch_idx"]
        correct_clean = checkpoint_test["correct_clean"]
        correct_adv = checkpoint_test["correct_adv"]
        print("************************************")
        print(f"Resuming testing from batch number {current_batch+2}")
        print("************************************")

    if not os.path.exists("%s" % (args.model_dir)):
        os.makedirs("%s" % (args.model_dir))

    for batch_idx, (data, target) in enumerate(test_loader):

        if batch_idx > current_batch:
            data, target = data.to(device), target.to(device)

            t0_epch_f = time()

            with torch.enable_grad():
                data_attack = gen_adv_data(
                    data,
                    target,
                    model,
                    optimizer,
                    args,
                )
            with torch.no_grad():
                # Forward pass on clean data
                output = model(data)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                is_correct_clean = pred.eq(target.view_as(pred)).sum().item()
                correct_clean += is_correct_clean
                # Forward pass on adversarial data
                output = model(data_attack)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                is_correct_adv = pred.eq(target.view_as(pred)).sum().item()
                correct_adv += is_correct_adv

            if "auto" in args.attack or "cw" in args.attack:
                checkpoint_test = {
                    "batch_idx": batch_idx,
                    "correct_clean": correct_clean,
                    "correct_adv": correct_adv,
                }
                torch.save(
                    checkpoint_test,
                    str(
                        "%s/%s_current_whitetest_checkpoint.pth"
                        % (args.model_dir, args.attack)
                    ),
                )

            T_epch_f = time() - t0_epch_f
            convert_seconds(T_epch_f)

            print(
                "[BATCH ({}) ({:.0f}%)]\tClean Accu: {:.2f} - Robust Accu: {:.2f}".format(
                    batch_idx + 1,
                    100.0 * batch_idx / len(test_loader),
                    100 * (is_correct_clean / args.test_batch_size),
                    100 * (is_correct_adv / args.test_batch_size),
                )
            )

    accuracy_clean = 100.0 * correct_clean / len(test_loader.dataset)
    accuracy_robust = 100.0 * correct_adv / len(test_loader.dataset)
    
    print("Clean Accuracy: ", accuracy_clean)
    print("Robust Accuracy: ", accuracy_robust)
    
    file_name = "results/results_%s_%s_%s_v01.txt" % (
        args.arch,
        args.data_name,
        args.which_model,
    )
    with open(file_name, "a") as file:
        file.write(
            f"ATTACK: {args.attack}-{args.num_steps} white-box *** METHOD: {args.method}\n"
        )
        file.write(
            f"Clean Accuracy: {accuracy_clean} - and - Robust Accuracy: {accuracy_robust}\n\n"
        )

    return [accuracy_clean, accuracy_robust]


def testing_adv_blackbox(
    model, model_adv, device, test_loader, args, resume_test=False
):
    """
    evaluate model by black-box attack
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
    )
    model.eval()
    model_adv.eval()
    correct_clean = 0
    correct_adv = 0
    current_batch = -1

    if resume_test:
        checkpoint_test = torch.load(
            str(
                "%s/%s_current_blacktest_checkpoint.pth" % (args.model_dir, args.attack)
            )
        )
        current_batch = checkpoint_test["batch_idx"]
        correct_clean = checkpoint_test["correct_clean"]
        correct_adv = checkpoint_test["correct_adv"]
        print("************************************")
        print(f"Resuming testing from batch number {current_batch+2}")
        print("************************************")

    if not os.path.exists("%s" % (args.model_dir)):
        os.makedirs("%s" % (args.model_dir))

    for batch_idx, (data, target) in enumerate(test_loader):

        if batch_idx > current_batch:
            data, target = data.to(device), target.to(device)

            t0_epch_f = time()

            with torch.enable_grad():
                data_attack = gen_adv_data(
                    data,
                    target,
                    model_adv,
                    optimizer,
                    args,
                )
            with torch.no_grad():
                # Forward pass on clean data
                output = model(data)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                correct_clean += pred.eq(target.view_as(pred)).sum().item()
                # Forward pass on adversarial data
                output = model(data_attack)
                output = F.log_softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                correct_adv += pred.eq(target.view_as(pred)).sum().item()

            if "auto" in args.attack or "cw" in args.attack:
                checkpoint_test = {
                    "batch_idx": batch_idx,
                    "correct_clean": correct_clean,
                    "correct_adv": correct_adv,
                }
                torch.save(
                    checkpoint_test,
                    str(
                        "%s/%s_current_blacktest_checkpoint.pth"
                        % (args.model_dir, args.attack)
                    ),
                )

            T_epch_f = time() - t0_epch_f

            print(
                "[BATCH ({}) ({:.0f}%)]\tTime: {:.6f}".format(
                    batch_idx + 1,
                    100.0 * batch_idx / len(test_loader),
                    T_epch_f,
                )
            )

    accuracy_clean = 100.0 * correct_clean / len(test_loader.dataset)
    accuracy_robust = 100.0 * correct_adv / len(test_loader.dataset)

    print("Clean Accuracy: ", accuracy_clean)
    print("Robust Accuracy: ", accuracy_robust)
    
    file_name = "results/results_%s_%s_%s_v01.txt" % (
        args.arch,
        args.data_name,
        args.which_model,
    )
    with open(file_name, "a") as file:
        file.write(
            f"ATTACK: {args.attack}-{args.num_steps} black-box *** METHOD: {args.method}\n"
        )
        file.write(
            f"Clean Accuracy: {accuracy_clean} - and - Robust Accuracy: {accuracy_robust}\n\n"
        )

    return [accuracy_clean, accuracy_robust]
