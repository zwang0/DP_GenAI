import torch

def compute_norm_squared_full_derivative(model, x_t, t):
    """
    Computes the norm squared of the full derivative of model(x_t, t) with respect to t using the chain rule:
    || d model(x_t, t) / dt ||^2 = ||partial model(x_t, t) / partial t + (Jacobian wrt x_t) * model(x_t, t)||^2

    Args:
        model: The UNet model which outputs dx_t / dt.
        x_t: The input image at time t (shape: [batch_size, 3, 32, 32]).
        t: The time input, which has shape [batch_size, 1].

    Returns:
        norm_squared: A scalar tensor representing the norm squared of the full derivative for each batch element (shape: [batch_size]).
    """

    # Set the model to evaluation mode to disable dropout, batchnorm, etc.
    model.eval()

    # Ensure t requires gradient to compute the partial derivative wrt t
    t.requires_grad_(True)

    # Forward pass: Compute model's prediction at (x_t, t), which gives dx_t/dt
    pred = model(x_t, t)  # pred is dx_t/dt, with shape [B, 3, 32, 32]

    # 1. Compute the partial derivative of the model output with respect to t
    partial_model_wrt_t = torch.autograd.grad(
        outputs=pred, inputs=t, grad_outputs=torch.ones_like(pred), create_graph=False)[0]

    # 2. Compute the Jacobian of the model output with respect to x_t
    B = x_t.shape[0]  # Batch size

    # Flatten x_t and pred using reshape instead of view
    x_t_flat = x_t.reshape(B, -1)  # Shape [B, 3072]
    pred_flat = pred.reshape(B, -1)  # Shape [B, 3072]

    # Function to compute the output w.r.t. a flattened x_t
    def model_output_wrt_x_t(x_t_single, t_single):
        # Reshape x_t_single back to [3, 32, 32] and run it through the model
        x_t_single = x_t_single.reshape(1, 3, 32, 32)
        return model(x_t_single, t_single).reshape(-1)  # Flatten the output to [3072]

    # Compute the Jacobian for each sample in the batch
    jacobians = []
    for i in range(B):
        jacobian = torch.autograd.functional.jacobian(
            lambda x: model_output_wrt_x_t(x, t[i]), x_t_flat[i]
        )
        jacobians.append(jacobian)

    # Stack the Jacobians to form a [B, 3072, 3072] tensor
    jacobians = torch.stack(jacobians)  # Shape: [B, 3072, 3072]

    # 3. Compute the total derivative using the chain rule
    # d model(x_t, t) / dt = partial_model_wrt_t + (Jacobian wrt x_t) * pred_flat
    partial_model_wrt_t_flat = partial_model_wrt_t.reshape(B, -1)  # Shape: [B, 3072]
    total_derivative_flat = partial_model_wrt_t_flat + torch.bmm(jacobians, pred_flat.unsqueeze(-1)).squeeze(-1)

    # 4. Compute the norm squared of the flattened total derivative
    norm_squared = torch.sum(total_derivative_flat ** 2, dim=1)  # Shape: [B]

    return norm_squared


# Example usage:
# Assume you have a model and input data
# x_t = torch.randn(B, 3, 32, 32)  # Example input image (batch size B)
# t = torch.randn(B, 1)  # Example time input

# norm_squared = compute_norm_squared_full_derivative(model, x_t, t)
