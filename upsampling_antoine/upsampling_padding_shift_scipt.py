"""File containing the two main functions used to exctract features using DINOv3 and increasing the output's resolution"""
import torch
from torchvision.transforms import v2, Pad

def make_transform_dinov3(resize_size: int = 1024):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def extract_dinov3_features(model, image, normalize:bool=False):
    """Make a forward of the model to extract the features"""
    with torch.inference_mode():
        outputs = model(image.unsqueeze(0).to(model.device))
        last_hidden_states = outputs.last_hidden_state
    patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]

    if normalize:
        # Normalize the output's features
        patch_features_flat = patch_features_flat/(torch.norm(patch_features_flat, dim=-1, keepdim=True)+1e-5)
    return patch_features_flat # (B, (H*W)/p_s, 1024)

def extract_features_high(model, input_image, normalize:bool=False):
    """Make a forward of the model to extract the features"""
    padding_size = 8
    padding = Pad(padding_size)
    input_padded_img = padding(input_image)
    patch_size = model.config.patch_size

    # Extract the features of the two views
    height = input_image.shape[2]
    extracted_features = extract_dinov3_features(model, input_image, normalize).reshape(1, height//patch_size, height//patch_size, -1) # (1, 64, 64, 1024)
    extracted_features_shifted = extract_dinov3_features(model, input_padded_img, normalize).reshape(1, height//patch_size + 1, height//patch_size + 1, -1) # (1, 65, 65, 1024)

    extracted_features = extracted_features.permute(0, 3, 1, 2)
    extracted_features_shifted = extracted_features_shifted.permute(0, 3, 1, 2)

    # Duplicate the features via nearest neighbor interpolation to match the final resolution
    features_up = torch.nn.functional.interpolate(extracted_features, scale_factor=2, mode='nearest')
    features_up_shifted = torch.nn.functional.interpolate(extracted_features_shifted, scale_factor=2, mode='nearest')

    # Take only the center features
    features_up_shifted = features_up_shifted[:, :, 1:-1, 1:-1]

    # Gather the features to perform the average pixel-wise
    all_features = torch.cat((features_up, features_up_shifted), dim=0).permute(0, 2, 3, 1) # (2, 128, 128, 1024)
    features_up = all_features.mean(dim=0, keepdim=True)
    features_flat_up = features_up.flatten(1, 2)
    return features_flat_up