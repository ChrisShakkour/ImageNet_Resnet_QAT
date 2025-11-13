
import torch
from .logger import *
import psutil, os

def model_precision_and_memory_report(model, logger):
    """
    Logs a detailed report of all parameters in a PyTorch model:
    - Name
    - Shape
    - Dtype (precision)
    - Memory footprint per parameter
    - Total memory usage

    Args:
        model (torch.nn.Module): The model to analyze.
        logger (logging.Logger): A configured logger to write the report.
    """
    total_params = 0
    total_memory = 0

    header = f"{'Parameter':40} {'Shape':25} {'Dtype':15} {'Memory':>10}"
    separator = "-" * len(header)
    logger.info(header)
    logger.info(separator)

    for name, param in model.named_parameters():
        num_params = param.numel()
        dtype = param.dtype
        mem_bytes = num_params * torch.finfo(dtype).bits // 8
        total_params += num_params
        total_memory += mem_bytes

        # Format memory nicely
        if mem_bytes < 1024:
            mem_str = f"{mem_bytes} B"
        elif mem_bytes < 1024 ** 2:
            mem_str = f"{mem_bytes / 1024:.2f} KB"
        else:
            mem_str = f"{mem_bytes / (1024 ** 2):.2f} MB"

        logger.info(f"{name:40} {str(tuple(param.shape)):25} {str(dtype):15} {mem_str:>10}")

    logger.info(separator)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Total memory: {total_memory / (1024 ** 2):.2f} MB\n")


def tensor_nbytes(tensor: torch.Tensor):
    """Return tensor memory footprint in bytes."""
    try:
        if tensor.dtype.is_floating_point:
            bits = torch.finfo(tensor.dtype).bits
        else:
            bits = torch.iinfo(tensor.dtype).bits
    except Exception:
        # Fallback for bool or unsupported dtypes
        bits = 8
    return tensor.numel() * bits // 8


def model_memory_report(model, logger=None):
    """
    Reports where model parameters and buffers are stored (CPU/GPU)
    and their memory footprint.
    """
    device_summary = {}
    total_memory = 0

    for name, param in model.named_parameters():
        device = str(param.device)
        mem_bytes = tensor_nbytes(param) // 8
        device_summary.setdefault(device, 0)
        device_summary[device] += mem_bytes
        total_memory += mem_bytes

    for name, buf in model.named_buffers():
        device = str(buf.device)
        mem_bytes = tensor_nbytes(param) // 8
        device_summary.setdefault(device, 0)
        device_summary[device] += mem_bytes
        total_memory += mem_bytes

    for device, mem_bytes in device_summary.items():
        mem_str = f"{mem_bytes / (1024**2):.2f} MB"
        msg = f"Device: {device:10}: Model tensors: {mem_str}"
        (logger.info(msg) if logger else print(msg))

    total_str = f"{total_memory / (1024**2):.2f} MB"
    (logger.info(f"Total model memory (params + buffers): {total_str}")
    if logger else print(f"Total model memory (params + buffers): {total_str}"))


def cpu_memory_report(logger=None):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    msg = f"CPU memory in use: {mem_mb:.2f} MB"
    (logger.info(msg) if logger else print(msg))


def gpu_runtime_report(device=0, logger=None):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    msg = f"GPU {device}: allocated {allocated:.2f} MB, reserved {reserved:.2f} MB"
    (logger.info(msg) if logger else print(msg))


def main():
    # Example usage
    import logging
    from torchvision.models import resnet18

    # Configure logger
    logger = init_logger("INFO")

    # Create a sample model
    model = resnet18()
    # Generate the report
    model_precision_and_memory_report(model, logger)
    # report memory by device
    model_memory_report(model, logger)
    
    # cut the model in half precision and report again
    model = model.half()    
    model_precision_and_memory_report(model, logger)
    model_memory_report(model, logger)


if __name__ == "__main__":
    main()