from huggingface_hub import HfApi
import psutil
import torch


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def print_hf_file_sizes(repo_id, repo_type='model'):
    api = HfApi()
    if repo_type == "model":
        dataset_info = api.model_info(repo_id=repo_id, files_metadata=True)
    elif repo_type == "dataset":
        dataset_info = api.dataset_info(repo_id=repo_id, files_metadata=True)

    total_size_bytes = 0  
    print(f"File sizes for dataset '{repo_id}':\n")  
    for sibling in dataset_info.siblings:  
        filename = sibling.rfilename  
        size_in_bytes = sibling.size or 0  
        total_size_bytes += size_in_bytes  
        size_mb = size_in_bytes / (1024 * 1024)  
        print(f"  {filename}: {size_mb:.2f} MiB")  

    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\nTotal size of {repo_type} {repo_id}: {total_size_mb:.2f} MiB or {total_size_mb/1000:.2f} GB")


def print_system_info():
    """
    Prints system information, including CUDA availability, GPU memory usage, and system RAM usage.
    """
    print("\n=== System & Memory Info ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device = torch.cuda.current_device()
        print(f"Using CUDA Device: {device} ({torch.cuda.get_device_name(device)})")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"GPU Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA is not available, running on CPU.")

    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / 1e9:.2f} GB")
    print(f"Available RAM: {ram.available / 1e9:.2f} GB")
    print(f"RAM Used: {ram.used / 1e9:.2f} GB ({ram.percent}%)")
    print("===========================\n")

