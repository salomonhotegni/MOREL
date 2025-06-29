"""
This file contains the main model class of MOREL for classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class morelnet(nn.Module):
    """
    Main model class for classification tasks.
    The original model's encoder (the model without its final layer) is passed
    as `mod_encoder` and the classifier head (the final layer) as `head_classifier`.
    """
    def __init__(self, mod_encoder, 
                 head_classifier, 
                 embed_dim = 128, 
                 num_att_heads = 2,
                 dropout = 0.0):
        super(morelnet, self).__init__()

        self.embed_dim = embed_dim
        self.num_att_heads = num_att_heads
        self.dropout = dropout

        self.feat_dim = head_classifier.in_features

        self.mod_encoder = mod_encoder
        self.head_classifier = head_classifier

        self.feat_layer = nn.Linear(self.feat_dim, self.embed_dim)
        self.pre_norm = nn.LayerNorm(self.embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_att_heads,
            dropout=self.dropout,
            batch_first=True,
        )

    def forward(self, x_in, training=False):
        """
        Args:
            x_in: Input tensor.
            training: Boolean indicating whether the model is in training mode.
        Returns:
            if training:
                pred_class: Predicted class probabilities.
                x: Encoded features (if training).
            else:
                pred_class: Predicted class probabilities.
        """

        x = self.mod_encoder(x_in)
        pred_class = self.head_classifier(x)
        if training:
            return pred_class, x
        else:
            return pred_class

    def shared_parameters(self):
        return self.mod_encoder.parameters()

    def get_emb_space_feats(self, enc_out, targets):
        """
        Get the embedded space features for the given encoded output and targets.
        """
        # Projecting the features to a lower-dimensional space
        x = self.feat_layer(enc_out)
        if targets is None:
            raise ValueError("`targets` must be given!")
        
        # Grouping features for each class
        unique_elmts = torch.unique(targets)
        indices_list = [
            torch.where(targets == element)[0].tolist() for element in unique_elmts
        ]
        class_feats = [x[indices] for indices in indices_list]

        # Class-Adaptive Multi-Head Attention
        attn_output = []
        attn_weights = []
        for i in range(len(unique_elmts)):
            x_norm_i = self.pre_norm(class_feats[i])

            attn_output_i, attn_weights_i = self.attention(x_norm_i, x_norm_i, x_norm_i)
            attn_output_i = class_feats[i] + attn_output_i
            attn_output.append(attn_output_i)
            attn_weights.append(attn_weights_i)

        return x, indices_list, attn_output
