import torch


'''
Loss functions for depth completion
'''
EPSILON = 1e-8

def l1_loss(src, tgt, w=None, normalize=False):
    '''
    Computes l1 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image (output depth)
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image (gt)
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        normalize : [bool]
            if normalize : normalized l1 loss 
            else : plain l1 loss  
    Returns:
        float : mean l1 loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)

    loss_func = torch.nn.L1Loss(reduction='none')
    loss = loss_func(src, tgt)
    if normalize:
        loss = loss / (torch.abs(tgt) + EPSILON)
    
    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])

    return torch.mean(loss)

def l2_loss(src, tgt, w=None, normalize=False):
    '''
    Computes l2 loss

    Arg(s):
        src : torch.Tensor[float32]
            N x 3 x H x W source image (output depth)
        tgt : torch.Tensor[float32]
            N x 3 x H x W target image (gt)
        w : torch.Tensor[float32]
            N x 1 x H x W weights
        normalize : [bool]
            if normalize : normalized l2 loss 
            else : plain l2 loss  
    Returns:
        float : mean l2 loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)


    loss_func = torch.nn.MSELoss(reduction='none')
    loss = loss_func(src, tgt)
    if normalize:
        loss = loss / (torch.pow(tgt, 2) + EPSILON)
        
    loss = torch.sum(w * loss, dim=[1, 2, 3]) / torch.sum(w, dim=[1, 2, 3])
    
    return torch.mean(loss)


def mse_focal(src, tgt, epoch=0, w=None):
    '''
    Computes Focal MSE Loss from "Sparse and Noisy LiDAR Completion with RGB Guidance and Uncertainty"
    Paper: https://arxiv.org/abs/1902.05356

    Arg(s):
        src: torch.Tensor[float32]
            N x 3 x H x W source array
        tgt: torch.Tensor[float32]
            N x 3 x H x W target array
        epoch: int
            current epoch of training
        w : torch.Tensor[float32]
            N x 1 x H x W weights
    Returns:
        float: mean loss across batch
    '''

    if w is None:
        w = torch.ones_like(src)

    if epoch == 0:
        return l2_loss(src, tgt, w)
    else:
        diff = w * (src - tgt)
        diff_sq = torch.square(diff)

        scaled_diff = 1 + 0.05 * epoch * diff
        loss = torch.mean(diff_sq + scaled_diff)

        return loss

def smoothness_loss_func(predict, image, w=None):
    '''
    Computes the local smoothness loss

    Arg(s):
        predict : torch.Tensor[float32]
            N x 1 x H x W predictions
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        weight : torch.Tensor[float32]
            N x 1 x H x W binary mask
    Returns:
        torch.Tensor[float32] : local smoothness loss
    '''

    # Add weights parameter (scalar)
    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    # Create edge awareness weights
    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))
    
    if w is not None:
        w_x = w[:,:,:,:-1]
        w_y = w[:,:,:-1,:]
    
        smoothness_x = torch.sum(w_x * weights_x * torch.abs(predict_dx)) / torch.sum(w_x)
        smoothness_y = torch.sum(w_y * weights_y * torch.abs(predict_dy)) / torch.sum(w_y)
    else:
        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
        
    return smoothness_x + smoothness_y


'''
Helper functions for constructing loss functions
'''
def gradient_yx(T):
    '''
    Computes gradients in the y and x directions

    Arg(s):
        T : torch.Tensor[float32]
            N x C x H x W tensor
    Returns:
        torch.Tensor[float32] : gradients in y direction
        torch.Tensor[float32] : gradients in x direction
    '''

    dx = T[:, :, :, :-1] - T[:, :, :, 1:]
    dy = T[:, :, :-1, :] - T[:, :, 1:, :]
    return dy, dx
