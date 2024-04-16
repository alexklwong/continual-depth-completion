import torch

# TODO: Add losses here as functions (e.g., EWC, LWF)

def compute_fisher(fisher_info, params, normalization):
    '''
    Calculate the fisher information matrix for a model's parameters on a task

    Arg(s):

    Returns:
        Fisher information matrix
    '''

    for idx, param in enumerate(params):
        if param.grad is not None:
            fisher_info[idx] += param.grad.data ** 2 / normalization

    return fisher_info


def ewc_loss(current_params, frozen_params, fisher_info, lambda_ewc):
    '''
    Calculate the ewc loss

    Arg(s):
        current_params : list[torch.Tensor[float32]]
        froze_params : list[torch.Tensor[float32]]
    Returns:

    '''
    loss = 0.0

    for curr, old, fisher in zip(current_params, frozen_params, fisher_info):
        loss += torch.sum(fisher * (old - curr)**2)

    return (lambda_ewc / 2) * loss