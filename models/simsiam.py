import torch
import torch.nn as nn
from lightly.models.simsiam import _prediction_mlp, _projection_mlp, SimSiam


class MaskSimSiam(SimSiam):
    def __init__(self,
                 backbone: nn.Module,
                 aux_num_mlp_layers: int = 2,
                 aux_out_dim: int = 128,
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 512,
                 out_dim: int = 2048,
                 num_mlp_layers: int = 3):

        super(MaskSimSiam, self).__init__(
            backbone = backbone,
            num_ftrs = num_ftrs,
            proj_hidden_dim = proj_hidden_dim,
            pred_hidden_dim = pred_hidden_dim,
            out_dim = out_dim,
            num_mlp_layers = num_mlp_layers
        )

        self.projection_mlp2 = \
            _projection_mlp(num_ftrs, proj_hidden_dim, aux_out_dim, aux_num_mlp_layers)

        self.prediction_mlp2 = \
            _prediction_mlp(aux_out_dim, pred_hidden_dim, aux_out_dim)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                masks: torch.Tensor = None,
                return_features: bool = False):
        """Forward pass through SimSiam.

        Extracts features with the backbone and applies the projection
        head and prediction head to the output space. If both x0 and x1 are not
        None, both will be passed through the backbone, projection, and
        prediction head. If x1 is None, only x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output prediction and projection of x0 and (if x1 is not None)
            the output prediction and projection of x1. If return_features is
            True, the output for each x is a tuple (out, f) where f are the
            features before the projection head.

        """
        f0 = self.backbone(x0).squeeze()
        if masks is not None:
            f0 = f0 * masks[None, :]
        z0 = self.projection_mlp(f0)
        p0 = self.prediction_mlp(z0)

        out0 = (z0, p0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None and masks is None:
            return out0

        f1 = self.backbone(x1).squeeze()
        if masks is not None:
            f1 = f1 * masks[None, :]
        z1 = self.projection_mlp(f1)
        p1 = self.prediction_mlp(z1)

        out1 = (z1, p1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        if masks is not None:
            z0_cl = self.projection_mlp2(f0)
            p0_cl = self.prediction_mlp2(z0_cl)

            out0_cl = (z0_cl, p0_cl)

            # append features if requested
            if return_features:
                out0_cl = (out0_cl, f0)

            if x1 is None:
                return out0_cl

            z1_cl = self.projection_mlp2(f1)
            p1_cl = self.prediction_mlp2(z1_cl)

            out1_cl = (z1_cl, p1_cl)

            # append features if requested
            if return_features:
                out1_cl = (out1_cl, f1)
            return out0, out1, out0_cl, out1_cl

        return out0, out1