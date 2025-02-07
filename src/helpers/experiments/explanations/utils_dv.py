"""This module adapted from the awesome open-source library: https://github.com/Nguyen-Hoa/Activation-Maximization to fit different channels."""

import torch
import numpy as np
import gc
from torchvision import transforms
from scipy.ndimage import gaussian_filter


def register_final_layer_hook(
    model: torch.nn.Module, activation_dict: dict, verbose: bool = False
):

    def hook(module, input, output):
        layer_name = "final_layer_output"
        activation_dict[layer_name] = output

    def find_last_module(module):
        children = list(module.children())
        if not children:
            return module
        else:
            return find_last_module(children[-1])

    final_module = find_last_module(model)
    if final_module is not None:
        final_module.register_forward_hook(hook)
        if verbose:
            print(f"Hook registered on final module: {final_module.__class__.__name__}")
        return "final_layer_output"
    else:
        if verbose:
            print("No suitable final module found.")
        return None


def layer_hook(act_dict, layer_name):
    """
    Create a hook into target layer
        Example to hook into classifier 6 of Alexnet:
            alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
    """

    def hook(module, input_img, output):
        act_dict[layer_name] = output

    return hook


def abs_contrib_crop(img, threshold=0):
    """
    Reguarlizer, crop by absolute value of pixel contribution
    """

    abs_img = torch.abs(img)
    smalls = abs_img < np.percentile(abs_img, threshold)

    return img - img * smalls


def norm_crop(img, threshold=0):
    """
    Regularizer, crop if norm of pixel values below threshold
    """

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Dynamically adjust the tiling based on the input_img tensor's channel count
    num_channels = img.shape[0]
    smalls = norm < np.percentile(norm, threshold)
    smalls = np.tile(smalls, (num_channels, 1, 1))

    crop = img - img * smalls
    return crop  # torch.from_numpy(crop)


def dv(
    network,
    input_img,
    layer_activation,
    layer_name,
    unit,
    steps,
    alpha=torch.tensor(100),
    # generate_gif=False,
    # path_to_gif="./",
    L2_Decay=True,
    theta_decay=0.1,
    Gaussian_Blur=True,
    theta_every=4,
    theta_width=1,
    verbose=False,
    Norm_Crop=True,
    theta_n_crop=30,
    Contrib_Crop=True,
    theta_c_crop=30,
    device: str = "cpu",
):
    """
    Optimizing Loop
        Dev: maximize layer vs neuron
    """

    network.to(device)

    best_activation = -float("inf")
    best_img = input_img

    # print(input_img.shape)
    if len(input_img.shape) == 3:
        input_img = input_img.unsqueeze(0)

    for k in range(steps):

        input_img.retain_grad()  # non-leaf tensor
        # network.zero_grad()

        # Propogate image through network,
        # then access activation of target layer
        network(input_img.to(device))
        layer_out = layer_activation[layer_name]

        # compute gradients w.r.t. target unit,
        # then access the gradient of input_img (image) w.r.t. target unit (neuron)
        layer_out[0][unit].backward(retain_graph=True)
        img_grad = input_img.grad

        # Gradient Step
        input_img = torch.add(input_img, torch.mul(img_grad, alpha))

        # Rgularization does not contribute towards gradient
        """
        DEV:
            Detach input_img here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input_img = torch.mul(input_img, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every == 0:
                temp = input_img.squeeze(0)
                # If temp is on GPU.
                if temp.is_cuda:
                    temp = temp.cpu()
                temp = temp.detach().numpy()
                num_channels = temp.shape[0]
                for channel in range(num_channels):
                    cimg = gaussian_filter(temp[channel], theta_width)
                    temp[channel] = cimg
                temp = torch.from_numpy(temp)
                input_img = temp.unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                input_img = norm_crop(
                    input_img.detach().squeeze(0), threshold=theta_n_crop
                )
                input_img = input_img.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                input_img = abs_contrib_crop(
                    input_img.detach().squeeze(0), threshold=theta_c_crop
                )
                input_img = input_img.unsqueeze(0)

        input_img.requires_grad_(True)

        if verbose:
            print("step: ", k, "activation: ", layer_out[0][unit])

        # if generate_gif:
        #    frame = input_img.detach().squeeze(0)
        #    frame = image_converter(frame)
        #    frame = frame * 255
        #    cv2.imwrite(path_to_gif + str(k) + ".jpg", frame)

        # Keep highest activation
        if best_activation < layer_out[0][unit]:
            best_activation = layer_out[0][unit]
            best_img = input_img

    # print(best_img.shape)
    if len(best_img.shape) == 4:
        best_img = best_img.squeeze(0)
    # print(best_img.shape)

    torch.cuda.empty_cache()
    gc.collect()

    return best_img
