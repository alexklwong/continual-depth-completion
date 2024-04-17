import torch

# TODO: Add losses here as functions (e.g., EWC, LWF)


def ewc_loss(current_params, frozen_params, fisher_info, lambda_ewc):
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

    for curr, old, fisher in zip(current_params, frozen_params, fisher_info):
        loss += torch.sum(fisher * (old - curr)**2)

    return (lambda_ewc / 2) * loss