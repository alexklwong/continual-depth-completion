import torch
import torch.nn.functional as F


def dominc_loss(queries, key_idx, key_list, lambda_dominc):
    '''
    Calculate the loss between query/key and between keys in the selector key list

    Args:
        queries : torch.Tensor[float32]
            Query vector [N, C]
        key_idx : int
            Index of the key in the key list
        key_list : torch.Tensor[float32]
            Key list K x [N]

        lambda_dominc : float
            Domain incremental loss weight
    '''
    loss = 0.0
    selected_key = key_list[key_idx]
    cosine_sim_qk = F.cosine_similarity(queries, selected_key.transpose(-2,-1), dim=1)
    loss_qk = 1 - cosine_sim_qk.mean()
    loss += loss_qk

    loss_kk = 0.0
    for i in range(len(key_list)):
        if i != key_idx:
            cosine_sim_kk = F.cosine_similarity(selected_key, key_list[i])  # should be a scalar!
            loss_kk += cosine_sim_kk
    if len(key_list) > 1:
        loss += loss_kk / (len(key_list) - 1)  # Normalize by number of key pools

    print("Domain Incremental Loss: ", loss)
    return lambda_dominc * loss


def ewc_loss(current_parameters, frozen_parameters, fisher_info, lambda_ewc):
    '''
    Calculate the ewc loss

    Arg(s):
        current_params : list[torch.Tensor[float32]]
        froze_params : list[torch.Tensor[float32]]
    Returns:
        loss : torch.Tensor[float32]
            EWC loss
    '''
    loss = 0.0

    for curr, old, fisher in zip(current_parameters, frozen_parameters, fisher_info):
        loss += torch.sum(fisher * (old - curr)**2)

    return (lambda_ewc / 2) * loss

def lwf_loss(output_depth0, output_frozen_depth0, lambda_lwf):
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
    # print(f"Type of output_depth0: {type(output_depth0)}, expected torch.Tensor")
    # print(f"Type of output_frozen_depth0: {type(output_frozen_depth0)}, expected torch.Tensor")

    if isinstance(output_depth0, list):
        output_depth0 = torch.cat(output_depth0, dim=0)

    if isinstance(output_frozen_depth0, list):
        output_frozen_depth0 = torch.cat(output_frozen_depth0, dim=0)

    if output_depth0.shape[0] != output_frozen_depth0.shape[0]:
        raise ValueError("Mismatch in batch size of output_depth0 and output_frozen_depth0")

    mse_loss = F.mse_loss(output_depth0, output_frozen_depth0)

    # Weight the loss by lambda_lwf
    combined_loss = lambda_lwf * mse_loss

    return combined_loss