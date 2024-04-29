import torch
import torch.nn.functional as F

#Before updating the model with the new task data, you run the new task data through the model (as it was trained on the old tasks) and record its predictions. These predictions serve as a stand-in for the actual ground truth of the old tasks, which you do not have. 
#During training on the new task, you aim to find a set of parameters for the model that not only performs well on the new task but also produces similar predictions to the recorded predictions by the frozen model. 
def compute_loss(self,
                output_depth0,
                output_frozen_depth0,
                lambda_lwf):
    '''
    Compute the LWF loss for unsupervised learning scenarios based on depth prediction.

    Args:
        output_depth0 : torch.Tensor
            current model's output depth maps [N, 1, H, W].
        output_frozen_depth0 : torch.Tensor[float32]
                frozen model's output depth maps N x 1 x H x W 
        lambda_lwf : float
            Regularization weight for the LWF loss component.

    Returns:
        torch.Tensor : The computed LWF loss, scaled by lambda_lwf\.
    '''

    # lwf LOSS: the difference between current and frozen model outputs. (original paper uses knoweldge distillation loss for classification)
    lwf_loss = F.mse_loss(output_depth0, output_frozen_depth0)
    
    # Weighted sum of the current depth loss and the LWF loss.
    combined_loss = lambda_lwf * lwf_loss
    
    return combined_loss

