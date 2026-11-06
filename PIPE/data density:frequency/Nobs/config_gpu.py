import torch

def config_gpu(gpu_flg):
    
    if torch.backends.mps.is_available():
        print("mps")
        device = torch.device("mps")
        return device
        
    
    if not torch.cuda.is_available():
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")
        return device
    
    if gpu_flg == 0:
        print("gpu_flg:", gpu_flg)
        
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device

    elif gpu_flg == 1:
        print("gpu_flg:", gpu_flg)
        
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)} with memory growth")
        return device

    elif gpu_flg == 2:
        print("gpu_flg:", gpu_flg)
        
        
        raise NotImplementedError("PyTorch does not support creating virtual GPUs.")

    else:
        raise NotImplementedError(">>>>> gpu_config")





