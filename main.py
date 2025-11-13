# Owned modules
from utils.args import *
from utils.logger import *
from utils.config_loader import *
from datasets.fake_dataset_loader import *
from datasets.imagenet_loader import *
from models.models import *
from utils.reports import *

# External modules
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
from torch.amp import autocast, GradScaler


def main():
    args = get_args()
    logger = init_logger(level=args.logging)
    cfg = load_yaml_config(args.config_file, logger)

    # Initialize data loader
    train_loader, val_loader = build_imagenet_loaders(cfg)
    logger.info('Dataset `%s` size:' % cfg.dataloader.dataset +
                '\n        Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader))
                )

    # Create the model
    model = create_model(cfg)
    # Report model precision and memory usage
    model_precision_and_memory_report(model, logger)

    # Insert quantizers into the model
    #modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    #model = quan.replace_module_by_names(model, modules_to_replace)

    # Enables data parallelism if multiple GPUs are available
    if cfg.device.gpu and not cfg.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=cfg.device.gpu)
    
    # Move model to the specified device
    model.to(cfg.device.type)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(cfg.device.type) #TODO: Why use this loss?

    # Define optimizer
    optimizer = t.optim.SGD(model.parameters(),
                            lr=cfg.optimizer.learning_rate,
                            momentum=cfg.optimizer.momentum,
                            weight_decay=cfg.optimizer.weight_decay)

    # Training loop
    start_epoch = 0
    epoch_losses = []  # Store average loss for each epoch
    scaler = GradScaler(device='cuda')

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            for epoch in range(start_epoch, cfg.epochs):
                # Set model to training mode
                model.train()
                batch_losses = []  # Store all batch losses for this epoch
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(cfg.device.type)
                    targets = targets.to(cfg.device.type)
                    
                    # Forward pass
                    with autocast(device_type='cuda', dtype=t.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Store batch loss
                    batch_losses.append(loss.item())
                    
                    #if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
                # Calculate and store average loss for this epoch
                epoch_loss = np.mean(batch_losses)
                epoch_losses.append(epoch_loss)
                #print(f'Epoch {epoch} average loss: {epoch_loss:.4f}')
                
                # Plot loss curve after each epoch
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(epoch_losses)), epoch_losses, 'b-', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss vs. Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig('loss_plot.png')
                plt.close()
    logger.info("\n" + prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    logger.info("\n" + prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    logger.info("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    logger.info("\n" + prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    #prof.export_chrome_trace("profiler_trace.json")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 🔍 PyTorch Profiler `sort_by` options
#
# | Key                      | Meaning                               | When to use                                       |
# | ------------------------ | ------------------------------------- | ------------------------------------------------- |
# | "cpu_time_total"         | Total CPU time (sum across all calls) | To see which ops take the most total CPU time     |
# | "cuda_time_total"        | Total CUDA GPU time                   | To see which ops dominate GPU compute time        |
# | "self_cpu_time_total"    | CPU time excluding child ops          | To find top-level Python ops that are bottlenecks |
# | "self_cuda_time_total"   | GPU time excluding child ops          | To see which CUDA kernels are directly expensive  |
# | "cpu_time"               | Average CPU time per call             | To find slowest operations individually           |
# | "cuda_time"              | Average CUDA time per call            | Same, but on GPU                                  |
# | "cpu_memory_usage"       | CPU memory used by the op             | To profile host-side memory footprint             |
# | "cuda_memory_usage"      | GPU memory used by the op             | To find which ops allocate GPU memory             |
# | "count"                  | Number of times the op was called     | To find very frequent small ops                   |
# | "input_shape"            | Input tensor shape                    | Useful when grouping by operation name            |
# | "name"                   | Operation name                        | To alphabetically organize ops                    |
#
# Example usage:
#   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# ---------------------------------------------------------------------------
