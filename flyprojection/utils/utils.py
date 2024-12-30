import importlib.util as implib
import numpy as np
import sys

# Import torch and kornia if available
try:
    import kornia
except ImportError:
    torch = None
    print("Warning: PyTorch and Kornia are not installed. 'kornia' method will not be available.")

def load_module_from_path(module_name, file_path):
    spec = implib.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for '{module_name}' at '{file_path}'")
    module = implib.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def tensor_to_numpy(tensor):
    """
    Convert a torch tensor to a NumPy array for display.

    Parameters:
        tensor (torch.Tensor): Tensor to convert. Shape [1, C, H, W].

    Returns:
        numpy.ndarray: Converted array. Shape [H, W, C], uint8.
    """
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()  # [H, W, C]
    array = (tensor * 255).astype(np.uint8)
    return array

def numpy_to_tensor(array):
    """
    Convert a NumPy array to a torch tensor.

    Parameters:
        array (numpy.ndarray): Array to convert. Shape [H, W, C] or [H, W].

    Returns:
        torch.Tensor: Converted tensor. Shape [1, C, H, W], float32.
    """
    tensor = torch.from_numpy(array.copy()).float() / 255.0  # Normalize to [0,1]
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.shape[2] == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    # move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def hex_to_rgb(value):
    """Convert hex color to RGB."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(tuple):
    """Convert RGB color to hex."""
    return '#%02x%02x%02x' % tuple
