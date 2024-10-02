import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

### MAIN MODEL

#######################################################
################### CLASSIFICATION ####################
#######################################################


class main_net_cl(nn.Module):
    def __init__(self, mod_encoder, head_classifier, args):
        super(main_net_cl, self).__init__()

        self.embed_dim = args.embed_dim
        self.num_att_heads = args.num_att_heads
        self.dropout = args.dropout

        self.feat_dim = args.feat_dim

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
        has_nan = torch.isnan(x_in).any()
        x = self.mod_encoder(x_in)

        pred_class = self.head_classifier(x)

        if training:
            return pred_class, x
        else:
            return pred_class

    def shared_parameters(self):
        return self.mod_encoder.parameters()

    def get_emb_space_feats(self, enc_out, targets):
        x = self.feat_layer(enc_out)

        if targets is None:
            raise ValueError("Targets must be given!")
        unique_elmts = torch.unique(targets)
        indices_list = [
            torch.where(targets == element)[0].tolist() for element in unique_elmts
        ]
        # Grouping features for each class
        class_feats = [x[indices] for indices in indices_list]

        attn_output = []
        attn_weights = []
        for i in range(len(unique_elmts)):
            x_norm_i = self.pre_norm(class_feats[i])

            attn_output_i, attn_weights_i = self.attention(x_norm_i, x_norm_i, x_norm_i)
            attn_output_i = class_feats[i] + attn_output_i
            attn_output.append(attn_output_i)
            attn_weights.append(attn_weights_i)

        return x, indices_list, attn_output


#######################################################
#######################################################
#######################################################
