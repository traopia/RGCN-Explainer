from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch 

# define utils functions to facilitate gpu 

def check_gpu_availability():
    # Check if CUDA is available
    print(f"Cuda is available: {torch.cuda.is_available()}")

def getting_device(gpu_prefence=True) -> torch.device:
    """
    This function gets the torch device to be used for computations, 
    based on the GPU preference specified by the user.
    """
    
    # If GPU is preferred and available, set device to CUDA
    if gpu_prefence and torch.cuda.is_available():
        device = torch.device('cuda')
    # If GPU is not preferred or not available, set device to CPU
    else: 
        device = torch.device("cpu")
    
    # Print the selected device
    print(f"Selected device: {device}")
    
    # Return the device
    return device

# Define a function to print GPU memory utilization
def print_gpu_utilization():
    # Initialize the PyNVML library
    nvmlInit()
    # Get a handle to the first GPU in the system
    handle = nvmlDeviceGetHandleByIndex(0)
    # Get information about the memory usage on the GPU
    info = nvmlDeviceGetMemoryInfo(handle)
    # Print the GPU memory usage in MB
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

# Define a function to print training summary information
def print_summary(result):
    # Print the total training time in seconds
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    # Print the number of training samples processed per second
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    # Print the GPU memory utilization
    print_gpu_utilization()


def clean_gpu():
    # Get current GPU memory usage
    print("BEFORE CLEANING:")
    print(f"Allocated: {cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {cuda.memory_cached() / 1024 ** 3:.2f} GB")
    print("\n")
    # Free up PyTorch and CUDA memory
    torch.cuda.empty_cache()
    cuda.empty_cache()
    
    # Run garbage collection to free up other memory
    gc.collect()
    
    # Get new GPU memory usage
    print("AFTER CLEANING:")
    print(f"Allocated: {cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {cuda.memory_cached() / 1024 ** 3:.2f} GB")    