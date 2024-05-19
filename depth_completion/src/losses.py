import torch
import torch.nn.functional as F

class lwf_loss:
    def __init__(self):
        """
        Initialize the Learning without Forgetting (LwF) loss class.
        """
        pass

    def __call__(self, output_depth0, output_frozen_depth0, lambda_lwf):
        """
        Compute the LwF loss for unsupervised learning scenarios based on depth prediction.

        Args:
            output_depth0 (torch.Tensor): Current model's output depth maps [N, 1, H, W].
            output_frozen_depth0 (torch.Tensor): Frozen model's output depth maps [N, 1, H, W].
            lambda_lwf (float): Regularization weight for the LwF loss component.

        Returns:
            torch.Tensor: The computed LwF loss, scaled by lambda_lwf.
        """
        # Calculate the MSE loss between current and frozen model outputs
        lwf_loss = F.mse_loss(output_depth0, output_frozen_depth0)

        # Weight the loss by lambda_lwf
        combined_loss = lambda_lwf * lwf_loss

        return combined_loss