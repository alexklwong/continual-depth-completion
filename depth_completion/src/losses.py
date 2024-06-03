import torch
import torch.nn.functional as F

def lwf_loss(output_depth0, output_frozen_depth0, lambda_lwf):
    """
    Compute the LwF loss for unsupervised learning scenarios based on depth prediction.

    Args:
        output_depth0 (list): Current model's output depth maps [N, 1, H, W].
        output_frozen_depth0 (list): Frozen model's output depth maps [N, 1, H, W].
        lambda_lwf (float): Regularization weight for the LwF loss component.

    Returns:
        torch.Tensor: The computed LwF loss, scaled by lambda_lwf.
    """
    # Calculate the MSE loss between current and frozen model outputs
    # print(f"Type of output_depth0: {type(output_depth0)}, expected torch.Tensor")
    # print(f"Type of output_frozen_depth0: {type(output_frozen_depth0)}, expected torch.Tensor")

    if isinstance(output_depth0, list):
        output_depth0 = output_depth0[0]

    if isinstance(output_frozen_depth0, list):
        output_frozen_depth0 = output_frozen_depth0[0]

    if output_depth0.shape[0] != output_frozen_depth0.shape[0]:
        raise ValueError("Mismatch in batch size of output_depth0 and output_frozen_depth0")

    mse_loss = F.mse_loss(output_depth0, output_frozen_depth0)

    # Weight the loss by lambda_lwf
    combined_loss = lambda_lwf * mse_loss

    return combined_loss