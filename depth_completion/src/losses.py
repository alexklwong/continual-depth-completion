import torch
import torch.nn.functional as F

#Before updating the model with the new task data, you run the new task data through the model (as it was trained on the old tasks) and record its predictions. These predictions serve as a stand-in for the actual ground truth of the old tasks, which you do not have. 
#During training on the new task, you aim to find a set of parameters for the model that not only performs well on the new task but also produces similar predictions to the recorded predictions by the frozen model. 
def compute_loss(self,
                output_depth0,
                image0,
                frozen_model,
                lambda_lwf):
    '''
    Compute the LWF loss for unsupervised learning scenarios based on depth prediction.

    Args:
        output_depth0 : torch.Tensor
            Tensor of the current model's output depth maps [N, 1, H, W].
        image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
        frozen_model : object
                instance of pretrained model frozen for loss computations
        lambda_lwf : float
            Regularization weight for the LWF loss component.

    Returns:
        torch.Tensor : The computed LWF loss, scaled by lambda_lwf\.
    '''
    #compute frozen model's output on current data
    frozen_model.eval()
    with torch.no_grad():
        frozen_model_output_depth0 = frozen_model(image0)
    # lwf LOSS: the difference between current and frozen model outputs. (original paper uses knoweldge distillation loss for classification)
    lwf_loss = F.mse_loss(output_depth0, frozen_model_output_depth0)
    
    # Weighted sum of the current depth loss and the LWF loss.
    combined_loss = lambda_lwf * lwf_loss
    
    return combined_loss

