"""
Implementation of the LOAT loss function for adversarial training.
Adapted from: https://github.com/TrustAI/LOAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def ALP_loss(logits_adv, logits):
    
    alp = torch.tensor(0, dtype=torch.float32)
    alp = alp.cuda()
    
    for idx_l in range(logits.size(0)):
        alp += (torch.sum((logits_adv - logits)**2))
    
    alp /= logits.size(0)

    return alp 

def LORE_v1(reg_type, logits_adv, logits, y, classes, device):

    correct = 0
    wrong = 0
    pred = torch.max(logits, dim=1)[1]
    probs = F.softmax(logits, dim=1)

    pred_adv = torch.max(logits_adv, dim=1)[1]
    probs_adv = F.softmax(logits_adv, dim=1)

    if reg_type == 'kl':
        fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct_prob = fluc_logit_correct_prob.to(device)
        fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong_prob = fluc_logit_wrong_prob.to(device)

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            logits_softmax_max = torch.tensor(0, dtype=torch.float32)
            
            for j in range(len(logits_softmax)):
                if j != y[idx_l].cpu().numpy():
                    logits_softmax_max = torch.max(logits_softmax[j], logits_softmax_max)
            
            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                fluc_logit_correct_prob -= torch.log(1 - logits_softmax_max)

            else:
                wrong += 1
                fluc_logit_wrong_prob -= torch.log(1 - logits_softmax_max)
                #fluc_logit_wrong_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))
        if correct == 0:
            fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
            fluc_logit_correct_prob = fluc_logit_correct_prob.to(device)
        else:
            fluc_logit_correct_prob /= correct
        if wrong == 0:
            fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
            fluc_logit_wrong_prob = fluc_logit_wrong_prob.to(device)
        else:
            fluc_logit_wrong_prob /= wrong
    
        return fluc_logit_correct_prob, fluc_logit_wrong_prob

    elif reg_type == 'mse':
        fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct = fluc_logit_correct.to(device)
        fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong = fluc_logit_wrong.to(device)
        
        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)

            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(classes-1))
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_correct += ((logits_softmax[idx] - mean_softmax)**2)
            else:
                wrong += 1
                mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(classes-1))
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_wrong += ((logits_softmax[idx] - mean_softmax)**2)

                #logits_adv_softmax = torch.softmax(logits_adv[idx_l], dim=0)
                #fluc_logit_wrong += (torch.sum((logits_adv_softmax - logits_softmax)**2))
        if correct == 0:
            fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
            fluc_logit_correct = fluc_logit_correct.to(device)
        else:
            fluc_logit_correct /= correct
        if wrong == 0:
            fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
            fluc_logit_wrong = fluc_logit_wrong.to(device)
        else:
            fluc_logit_wrong /= wrong
        return fluc_logit_correct, fluc_logit_wrong


def LORE(reg_type, logits_adv, logits, y, classes, device, choice='wrong'):

    correct = 0
    wrong = 0

    pred = torch.max(logits, dim=1)[1]
    pred_adv = torch.max(logits_adv, dim=1)[1]

    probs = F.softmax(logits, dim=1)
    probs_adv = F.softmax(logits_adv, dim=1)

    if reg_type == 'kl':
        fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct_prob = fluc_logit_correct_prob.to(device)
        fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong_prob = fluc_logit_wrong_prob.to(device)
        
        alp_prob = torch.tensor(0, dtype=torch.float32)
        alp_prob = alp_prob.to(device)

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            logits_softmax_max = torch.tensor(0, dtype=torch.float32)
            
            for j in range(len(logits_softmax)):
                if j != y[idx_l].cpu().numpy():
                    logits_softmax_max = torch.max(logits_softmax[j], logits_softmax_max)
            
            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                fluc_logit_correct_prob -= torch.log(1 - logits_softmax_max)
                if choice == 'correct':
                    alp_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))
            else:
                wrong += 1
                fluc_logit_wrong_prob -= torch.log(1 - logits_softmax_max)
                if choice == 'wrong':
                    alp_prob -= torch.log(1 - torch.abs(probs_adv[idx_l][pred_adv[idx_l]] - probs[idx_l][pred_adv[idx_l]]))
        
        if correct == 0:
            fluc_logit_correct_prob = torch.tensor(0, dtype=torch.float32)
            fluc_logit_correct_prob = fluc_logit_correct_prob.to(device)
            if choice == 'correct':
                alp_prob = torch.tensor(0, dtype=torch.float32)
                alp_prob = alp_prob.to(device)
        else:
            fluc_logit_correct_prob /= correct
            if choice == 'correct':
                alp_prob /= correct
            
        if wrong == 0:
            fluc_logit_wrong_prob = torch.tensor(0, dtype=torch.float32)
            fluc_logit_wrong_prob = fluc_logit_wrong_prob.to(device)
            if choice == 'wrong':
                alp_prob = torch.tensor(0, dtype=torch.float32)
                alp_prob = alp_prob.to(device)
        else:
            fluc_logit_wrong_prob /= wrong
            if choice == 'wrong':
                alp_prob /= wrong

        return fluc_logit_correct_prob, fluc_logit_wrong_prob, alp_prob

    elif reg_type == 'mse':
        fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
        fluc_logit_correct = fluc_logit_correct.to(device)
        fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
        fluc_logit_wrong = fluc_logit_wrong.to(device)

        alp = torch.tensor(0, dtype=torch.float32)
        alp = alp.to(device)

        for idx_l in range(y.size(0)):
            logits_softmax = torch.softmax(logits[idx_l], dim=0)
            mean_softmax = ((1 - logits_softmax[y[idx_l].cpu().numpy()])/(classes-1))
            logits_adv_softmax = torch.softmax(logits_adv[idx_l], dim=0)

            if pred[idx_l].cpu().numpy() == y[idx_l].cpu().numpy():
                correct += 1
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_correct += ((logits_softmax[idx] - mean_softmax)**2)
                if choice == 'correct':
                    alp += (torch.sum((logits_adv_softmax - logits_softmax)**2))

            else:
                wrong += 1
                for idx in range(len(logits_softmax)):
                    if idx != y[idx_l].cpu().numpy():
                        fluc_logit_wrong += ((logits_softmax[idx] - mean_softmax)**2)
                if choice == 'wrong':
                    alp += (torch.sum((logits_adv_softmax - logits_softmax)**2))
        
        if correct == 0:
            fluc_logit_correct = torch.tensor(0, dtype=torch.float32)
            fluc_logit_correct = fluc_logit_correct.to(device)
            if choice == 'correct':
                alp = torch.tensor(0, dtype=torch.float32)
                alp = alp.to(device)
        else:
            fluc_logit_correct /= correct
            if choice == 'correct':
                alp /= correct
                
        if wrong == 0:
            fluc_logit_wrong = torch.tensor(0, dtype=torch.float32)
            fluc_logit_wrong = fluc_logit_wrong.to(device)
            if choice == 'wrong':
                alp = torch.tensor(0, dtype=torch.float32)
                alp = alp.to(device)
        else:
            fluc_logit_wrong /= wrong
            if choice == 'wrong':
                alp /= wrong

        return fluc_logit_correct, fluc_logit_wrong, alp

def loat_loss(model,
              epoch,
              x_natural,
              y,
              reg,
              reg_type,
              optimizer,
              device,
              classes = 10,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf',
              theta=1.0,
              gamma=0.5,
              lot='LORE',
              extra_outputs=False):
    
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    pred = torch.max(logits, dim=1)[1]
    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    
    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    
    if not reg:
        loss = loss_adv + float(beta) * loss_robust
    else:
        if epoch <= 1:
            if lot == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1(reg_type, logits_adv, logits, y, classes, device)
                reg_val = fluc_correct - fluc_wrong
                loss = loss_adv + float(beta) * loss_robust + theta * reg_val
            elif lot == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE(reg_type, logits_adv, logits, y, classes, device, 'wrong')
                reg_val = fluc_correct - fluc_wrong
                loss = loss_adv + float(beta) * loss_robust + theta * reg_val + gamma * alp
        elif epoch >= 100:
            if lot == 'LORE_v1':
                fluc_correct, fluc_wrong = LORE_v1(reg_type, logits_adv, logits, y, classes, device)
                reg_val = fluc_wrong - fluc_correct
                loss = loss_adv + float(beta) * loss_robust + theta * reg_val
            elif lot == 'LORE':
                fluc_correct, fluc_wrong, alp = LORE(reg_type, logits_adv, logits, y, classes, device, 'correct')
                reg_val = fluc_wrong - fluc_correct                
                loss = loss_adv + float(beta) * loss_robust + theta * reg_val + gamma * alp
        else:
            loss = loss_adv + float(beta) * loss_robust
            
    if extra_outputs:
        return loss, x_adv
    else:
        return loss