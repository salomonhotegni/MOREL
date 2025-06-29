"""
This module implements our adversarial training framework (MOREL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from advermorel.losses.trades import trades_loss
from advermorel.losses.mart import mart_loss
from advermorel.losses.loat import loat_loss


def morel_loss(
    model,
    x_natural,
    y,
    optimizer,
    k,  # preference vector
    a,  # reference point
    gamma,  # augmentation coef
    accu_obj=None,
    alpha=1e-5,
    step_size=0.007,
    epsilon=0.031,
    perturb_steps=10,
    beta=6.0,
    distance="l_inf",
    extra_configs=None,
    extra_outputs=False,
    custom_accu_obj=None,
):
    """
    Compute the MOREL loss.
    Parameters:
        model: The model to train.
        x_natural: The natural input data.
        y: The labels for the input data.
        optimizer: The optimizer for the model.
        k: Preference vector.
        a: Reference point.
        gamma: Augmentation coefficient.
        accu_obj: The accuracy loss function (e.g., "trades", "mart", "loat").
        alpha: Weight for the Contrastive loss.
        step_size: Step size for adversarial perturbation.
        epsilon: Maximum perturbation.
        perturb_steps: Number of perturbation steps.
        beta: Weight for the robustness loss.
        distance: Distance metric for perturbation ("l_inf" or "l_2").
        extra_configs: Additional configurations for specific losses.
        extra_outputs: If True, return additional outputs.
        custom_accu_obj: Custom accuracy loss function if provided. It should accept the following parameters:
            model, x_natural, y, optimizer, and extra_configs, where extra_configs is a dictionary
            containing any additional configurations needed by the custom loss function.
            It should return a tuple of (loss for accuracy, adversarial batch).
    Returns:
        ConeScal_loss: The computed MOREL loss.
        orig_loss: Original losses (robust loss and clean loss).
        list_indiv_loss: List of individual losses (natural, robust, cosine similarity, contrastive).
    """
    # loss for accuracy (with a touch of robustness)
    if custom_accu_obj is None:
        if accu_obj is None:
            raise ValueError("Either accu_obj or custom_accu_obj must be provided.")
        elif "trades" in accu_obj:
            a_loss, x_adv = trades_loss(
                model,
                x_natural,
                y,
                optimizer,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                beta=beta,
                distance=distance,
                extra_outputs=True,
            )

        elif "mart" in accu_obj:
            a_loss, x_adv = mart_loss(
                model,
                x_natural,
                y,
                optimizer,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                beta=beta,
                distance=distance,
                extra_outputs=True,
            )
        elif "loat" in accu_obj:
            assert (extra_configs is not None), "Extra configs must be provided for LOAT loss."
            assert (extra_configs.get("epoch") is not None), "Epoch must be specified in extra configs."
            assert (extra_configs.get("reg") is not None), "Regularization term must be specified in extra configs."
            assert (extra_configs.get("reg_type") is not None), "Regularization type must be specified in extra configs."
            assert (extra_configs.get("device") is not None), "Device must be specified in extra configs."
            assert (extra_configs.get("theta") is not None), "Theta must be specified in extra configs."
            assert (extra_configs.get("gamma_loat") is not None), "Gamma for LOAT must be specified in extra configs."
            assert (extra_configs.get("lot") is not None), "Lot value must be specified in extra configs."
            a_loss, x_adv = loat_loss(model=model,
                                                epoch=extra_configs.get("epoch"),
                                                x_natural=x_natural,
                                                y=y,
                                                reg=extra_configs.get("reg"),
                                                reg_type=extra_configs.get("reg_type"),
                                                optimizer=optimizer,
                                                device=extra_configs.get("device"),
                                                step_size=step_size,
                                                epsilon=epsilon,
                                                perturb_steps=perturb_steps,
                                                beta=beta,
                                                theta=extra_configs.get("theta"),
                                                gamma=extra_configs.get("gamma_loat"),
                                                lot=extra_configs.get("lot"),
                                                extra_outputs=True)

        else:
            raise ValueError(f"Loss function {accu_obj} not implemented !")
        
    else:
        a_loss, x_adv = custom_accu_obj(
                model=model,
                x_natural=x_natural,
                y=y,
                optimizer=optimizer,
                extra_configs=extra_configs,
            )

    # Get Encoder features
    _, enc_feats = model(x_natural, training=True)
    _, enc_feats_adv = model(x_adv, training=True)

    # Get embedded features
    batch_feats, indices_list, attn_output = model.get_emb_space_feats(enc_feats, y)
    batch_feats_adv, _, attn_output_adv = model.get_emb_space_feats(enc_feats_adv, y)

    # Normalize lower dimensional features
    attn_output_n = torch.zeros_like(batch_feats)
    for indices, segment in zip(indices_list, attn_output):
        attn_output_n[indices] = segment
    attn_output_n = F.normalize(attn_output_n, dim=-1, p=2)

    attn_output_adv_n = torch.zeros_like(batch_feats)
    for indices, segment in zip(indices_list, attn_output_adv):
        attn_output_adv_n[indices] = segment
    attn_output_adv_n = F.normalize(attn_output_adv_n, dim=-1, p=2)

    # loss for robustness (based on features)
    criterion_cs = MultiPosConLoss(device="cuda")
    criterion_cos = cosine_similarity_loss
    ## Cosine Similarity Loss
    cos_loss = criterion_cos(batch_feats, batch_feats_adv)
    ## Contrastive Loss
    cs_loss = criterion_cs(attn_output_n, y)
    ## Morel Robust Loss
    f_loss = cos_loss + alpha * cs_loss

    # Compute Cone Scalarization Loss
    losses = torch.stack([f_loss, a_loss])
    rect_losses = [losses[0] - a[0], losses[1] - a[1]]
    augm_term = gamma * sum(rect_losses)
    ConeScal_loss = sum([k[0] * rect_losses[0], k[1] * rect_losses[1]]) + augm_term

    if not extra_outputs:
        return ConeScal_loss

    # Extra Outputs
    orig_loss = [f_loss, a_loss]
    orig_loss = (
        torch.cat([tens.reshape(1) for tens in orig_loss]).detach().cpu().reshape(-1, 1)
    )

    return  ConeScal_loss, orig_loss


def cosine_similarity_loss(orig_feats, new_feats):
    """
    Cosine similarity loss function.
    This function computes the cosine similarity between two sets of features.
    It returns a loss value that is 1 minus the average cosine similarity across the batch.
    Parameters:
        orig_feats: Original features (tensor).
        new_feats: New features (tensor).
    Returns:
        cosine_loss: The computed cosine similarity loss.
    """
    # Calculate cosine similarity between corresponding rows
    cos_sim = F.cosine_similarity(orig_feats, new_feats, dim=1)
    cosine_loss = 1 - cos_sim.mean()  # Averaging the similarity scores across the batch
    return cosine_loss


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, device, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        """
        Args:
            feats: Feature representations of the input data.
            labels: Corresponding labels for the input data.
        Returns:
            loss: Computed Multi-Positive Contrastive Loss.
        """
        # Compute the mask based on combined labels
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(self.device)

        # Create logits mask to exclude self-similarities
        logits_mask = torch.ones_like(mask) - torch.eye(
            mask.size(0), device=self.device
        )

        # Apply the mask
        mask *= logits_mask

        # Compute logits
        logits = torch.matmul(feats, feats.T) / self.temperature  #  (!)
        logits = logits - (1 - logits_mask) * 1e9

        # Stabilize logits by subtracting the maximum logit
        logits = self.stabilize_logits(logits)

        # Compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # Compute loss
        loss = self.compute_cross_entropy(p, logits)

        return loss

    def compute_cross_entropy(self, p, q):
        q = F.log_softmax(q, dim=-1)
        loss = torch.sum(p * q, dim=-1)
        return -loss.mean()

    def stabilize_logits(self, logits):
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        return logits
