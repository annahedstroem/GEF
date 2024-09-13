"""This module adapted from the awesome open-source library: https://github.com/serre-lab/Horama to fit different channels."""

import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.ops import roi_align

MACO_SPECTRUM_URL = (
    "https://storage.googleapis.com/serrelab/loupe/spectrums/imagenet_decorrelated.npy"
)
MACO_SPECTRUM_FILENAME = "spectrum_decorrelated.npy"


def cosine_similarity(tensor_a, tensor_b):
    # calculate cosine similarity
    norm_dims = list(range(1, len(tensor_a.shape)))
    tensor_a = torch.nn.functional.normalize(tensor_a.float(), dim=norm_dims)
    tensor_b = torch.nn.functional.normalize(tensor_b.float(), dim=norm_dims)
    return torch.sum(tensor_a * tensor_b, dim=norm_dims)


def dot_cossim(tensor_a, tensor_b, cossim_pow=2.0):
    # compute dot product scaled by cosine similarity
    cosim = torch.clamp(cosine_similarity(tensor_a, tensor_b), min=1e-1) ** cossim_pow
    dot = torch.sum(tensor_a * tensor_b)
    return dot * cosim


def standardize(tensor):
    # standardizes the tensor to have 0 mean and unit variance
    tensor = tensor - torch.mean(tensor)
    tensor = tensor / (torch.std(tensor) + 1e-4)
    return tensor


def recorrelate_colors(image, device):

    # tensor for color correlation svd square root
    color_correlation_svd_sqrt = torch.tensor(
        [
            [0.56282854, 0.58447580, 0.58447580],
            [0.19482528, 0.00000000, -0.19482528],
            [0.04329450, -0.10823626, 0.06494176],
        ],
        dtype=torch.float32,
    ).to(device)

    # recorrelates the colors of the images
    assert len(image.shape) == 3

    permuted_image = image.permute(1, 2, 0).contiguous()
    flat_image = permuted_image.view(-1, 3)

    recorrelated_image = torch.matmul(flat_image, color_correlation_svd_sqrt)
    recorrelated_image = recorrelated_image.view(permuted_image.shape).permute(2, 0, 1)

    return recorrelated_image


def optimization_step(
    objective_function,
    image,
    box_size,
    noise_level,
    number_of_crops_per_iteration,
    model_input_size,
    nr_channels,
):
    # performs an optimization step on the generated image
    assert box_size[1] >= box_size[0]
    assert len(image.shape) == 3

    device = image.device
    image.retain_grad()

    # generate random boxes
    x0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    y0 = 0.5 + torch.randn((number_of_crops_per_iteration,), device=device) * 0.15
    delta_x = (
        torch.rand((number_of_crops_per_iteration,), device=device)
        * (box_size[1] - box_size[0])
        + box_size[1]
    )
    delta_y = delta_x

    boxes = (
        torch.stack(
            [
                torch.zeros((number_of_crops_per_iteration,), device=device),
                x0 - delta_x * 0.5,
                y0 - delta_y * 0.5,
                x0 + delta_x * 0.5,
                y0 + delta_y * 0.5,
            ],
            dim=1,
        )
        * image.shape[1]
    )
    # print(f"boxes {boxes.shape}")

    cropped_and_resized_images = roi_align(
        image.unsqueeze(0), boxes, output_size=(model_input_size, model_input_size)
    ).squeeze(0)

    if nr_channels == 1:
        cropped_and_resized_images = cropped_and_resized_images.mean(axis=1).unsqueeze(
            1
        )

    # print(f"cropped_and_resized_images {cropped_and_resized_images.shape}")

    # add normal and uniform noise for better robustness
    cropped_and_resized_images.add_(
        torch.randn_like(cropped_and_resized_images) * noise_level
    )
    cropped_and_resized_images.add_(
        (torch.rand_like(cropped_and_resized_images) - 0.5) * noise_level
    )

    # print(f"cropped_and_resized_images {cropped_and_resized_images.shape}")
    # print(f"image {image.shape}")

    # compute the score and loss. # TOTDO. Maybe this breaks.
    # if nr_channels == 1:
    #    score = objective_function(cropped_and_resized_images)
    # else:
    score = objective_function(cropped_and_resized_images)

    loss = -score

    return loss, image


def fft_2d_freq(width, height):
    # calculate the 2D frequency grid for FFT
    freq_y = torch.fft.fftfreq(height).unsqueeze(1)

    cut_off = int(width % 2 == 1)
    freq_x = torch.fft.fftfreq(width)[: width // 2 + 1 + cut_off]

    return torch.sqrt(freq_x**2 + freq_y**2)


def get_fft_scale(width, height, decay_power=1.0):
    # generate the FFT scale based on the image size and decay power
    frequencies = fft_2d_freq(width, height)

    fft_scale = (
        1.0
        / torch.maximum(frequencies, torch.tensor(1.0 / max(width, height)))
        ** decay_power
    )
    fft_scale = fft_scale * torch.sqrt(torch.tensor(width * height).float())

    return fft_scale.to(torch.complex64)


def init_olah_buffer(width, height, nr_channels, std=1.0):
    # initialize the Olah buffer with a random spectrum
    spectrum_shape = (nr_channels, width, height // 2 + 1)
    random_spectrum = torch.complex(
        torch.randn(spectrum_shape) * std, torch.randn(spectrum_shape) * std
    )
    return random_spectrum


def fourier_preconditionner(
    spectrum, spectrum_scaler, values_range, nr_channels, device
):
    # precondition the Fourier spectrum and convert it to spatial domain
    if nr_channels == 1:
        assert spectrum.shape[0] == 1
    else:
        assert spectrum.shape[0] == 3

    spectrum = standardize(spectrum)
    spectrum = spectrum * spectrum_scaler

    spatial_image = torch.fft.irfft2(spectrum)
    spatial_image = standardize(spatial_image)

    if nr_channels == 3:
        color_recorrelated_image = recorrelate_colors(spatial_image, device)
        final_image = (
            torch.sigmoid(color_recorrelated_image)
            * (values_range[1] - values_range[0])
            + values_range[0]
        )
    else:  # skip recorrelation
        final_image = (
            torch.sigmoid(spatial_image) * (values_range[1] - values_range[0])
            + values_range[0]
        )
        final_image = final_image.mean(axis=0).unsqueeze(0)

    return final_image


def fourier(
    objective_function,
    decay_power=1.5,
    total_steps=1000,
    learning_rate=1.0,
    image_size=1280,
    model_input_size=224,
    noise=0.05,
    values_range=(-2.5, 2.5),
    crops_per_iteration=6,
    box_size=(0.20, 0.25),
    device="cuda",
    nr_channels=3,
):
    # perform the Olah optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]
    # Get the dims of image_size first dim

    spectrum = init_olah_buffer(image_size, image_size, nr_channels, std=1.0)
    spectrum_scaler = get_fft_scale(image_size, image_size, decay_power)

    spectrum = spectrum.to(device)
    spectrum.requires_grad = True
    spectrum_scaler = spectrum_scaler.to(device)

    optimizer = torch.optim.NAdam([spectrum], lr=learning_rate)
    # print("nr_channels", nr_channels)
    transparency_accumulator = torch.zeros((nr_channels, image_size, image_size)).to(
        device
    )

    for step in tqdm(range(total_steps)):
        optimizer.zero_grad()

        image = fourier_preconditionner(
            spectrum, spectrum_scaler, values_range, nr_channels, device
        )
        loss, img = optimization_step(
            objective_function,
            image,
            box_size,
            noise,
            crops_per_iteration,
            model_input_size,
            nr_channels,
        )
        loss.backward()
        if nr_channels == 1:
            transparency_accumulator += torch.abs(img.grad.mean(axis=0))
        else:
            transparency_accumulator += torch.abs(img.grad)
        optimizer.step()

    final_image = fourier_preconditionner(
        spectrum, spectrum_scaler, values_range, nr_channels, device
    )
    return final_image, transparency_accumulator


def init_maco_buffer(image_shape, nr_channels, std_deviation=1.0):
    # initialize the maco buffer with a random phase and a magnitude template
    spectrum_shape = (image_shape[0], image_shape[1] // 2 + 1)
    # generate random phase
    random_phase = (
        torch.randn(nr_channels, *spectrum_shape, dtype=torch.float32) * std_deviation
    )

    # download magnitude template if not exists
    if not os.path.isfile(MACO_SPECTRUM_FILENAME):
        download_url(MACO_SPECTRUM_URL, root=".", filename=MACO_SPECTRUM_FILENAME)

    # load and resize magnitude template
    magnitude = torch.tensor(
        np.load(MACO_SPECTRUM_FILENAME), dtype=torch.float32
    ).cuda()
    magnitude = F.interpolate(
        magnitude.unsqueeze(0),
        size=spectrum_shape,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]

    return magnitude, random_phase


def maco_preconditioner(magnitude_template, phase, values_range, nr_channels, device):
    # apply the maco preconditioner to generate spatial images from magnitude and phase
    # tfel: check why r exp^(j theta) give slighly diff results
    standardized_phase = standardize(phase)
    complex_spectrum = torch.complex(
        torch.cos(standardized_phase) * magnitude_template,
        torch.sin(standardized_phase) * magnitude_template,
    )

    # transform to spatial domain and standardize
    spatial_image = torch.fft.irfft2(complex_spectrum)
    spatial_image = standardize(spatial_image)

    # recorrelate colors and adjust value range
    if nr_channels == 3:
        color_recorrelated_image = recorrelate_colors(spatial_image, device)
        final_image = (
            torch.sigmoid(color_recorrelated_image)
            * (values_range[1] - values_range[0])
            + values_range[0]
        )
    else:  # skip recorrelation
        final_image = (
            torch.sigmoid(spatial_image) * (values_range[1] - values_range[0])
            + values_range[0]
        )
        final_image = final_image.mean(axis=0).unsqueeze(0)

    return final_image


def maco(
    objective_function,
    total_steps=1000,
    learning_rate=1.0,
    image_size=1280,
    model_input_size=224,
    noise=0.05,
    values_range=(-2.5, 2.5),
    crops_per_iteration=6,
    box_size=(0.20, 0.25),
    device="cuda",
    nr_channels=3,
):
    # perform the maco optimization process
    assert values_range[1] >= values_range[0]
    assert box_size[1] >= box_size[0]

    magnitude, phase = init_maco_buffer(
        (image_size, image_size), nr_channels=nr_channels, std_deviation=1.0
    )
    magnitude = magnitude.to(device)
    phase = phase.to(device)
    phase.requires_grad = True

    optimizer = torch.optim.NAdam([phase], lr=learning_rate)
    transparency_accumulator = torch.zeros((nr_channels, image_size, image_size)).to(
        device
    )

    for step in tqdm(range(total_steps)):
        optimizer.zero_grad()

        # preprocess and compute loss
        img = maco_preconditioner(magnitude, phase, values_range, nr_channels, device)
        loss, img = optimization_step(
            objective_function,
            img,
            box_size,
            noise,
            crops_per_iteration,
            model_input_size,
            nr_channels,
        )

        loss.backward()
        # get dy/dx to update transparency mask
        if nr_channels == 1:
            transparency_accumulator += torch.abs(img.grad.mean(axis=0))
        else:
            transparency_accumulator += torch.abs(img.grad)
        optimizer.step()

    final_image = maco_preconditioner(
        magnitude, phase, values_range, nr_channels, device
    )

    return final_image, transparency_accumulator
