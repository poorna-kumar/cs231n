import torch


def convert_to_np(obj, device):
    if torch.is_tensor(obj):
        data = obj.data
        # If running on GPU, we need to call .cpu() to copy the tensor to host memory
        # before we can convert it to numpy
        if 'cuda' in device.type:
            data = data.cpu()
        return data.numpy()

    return obj
