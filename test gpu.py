import os
import time
import torch
import torch.nn as nn


def get_env_info():
    info = {}
    info["torch_version"] = torch.__version__
    info["cuda_compiled"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count()
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    return info


def get_gpu_list():
    gpus = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        gpus.append({"index": i, "name": name, "capability": capability})
    return gpus


def pick_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_tensor_op(device):
    start = time.time()
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = torch.mm(a, b)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dur = time.time() - start
    return c.norm().item(), dur


def test_small_model(device):
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 128), nn.ReLU(), nn.Linear(128, 4))
    model.to(device)
    x = torch.randn(32, 3, 224, 224, device=device)
    start = time.time()
    y = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    dur = time.time() - start
    return y.mean().item(), dur


def format_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def get_memory(device):
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(idx)
    allocated = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    total = torch.cuda.get_device_properties(idx).total_memory
    return {"allocated": allocated, "reserved": reserved, "total": total}


def main():
    info = get_env_info()
    print(f"torch_version={info['torch_version']}")
    print(f"cuda_compiled={info['cuda_compiled']}")
    print(f"cudnn_version={info['cudnn_version']}")
    print(f"cuda_available={info['cuda_available']}")
    print(f"cuda_device_count={info['cuda_device_count']}")
    print(f"cuda_visible_devices={info['cuda_visible_devices']}")
    if info["cuda_available"] and info["cuda_device_count"] > 0:
        gpus = get_gpu_list()
        for g in gpus:
            print(f"gpu_index={g['index']} gpu_name={g['name']} capability={g['capability']}")
    device = pick_device()
    print(f"selected_device={device}")
    mem = get_memory(device)
    if mem is not None:
        print(f"memory_allocated={format_bytes(mem['allocated'])}")
        print(f"memory_reserved={format_bytes(mem['reserved'])}")
        print(f"memory_total={format_bytes(mem['total'])}")
    try:
        norm_val, dur_matmul = test_tensor_op(device)
        print(f"tensor_op_ok=True norm={norm_val:.4f} duration_s={dur_matmul:.4f}")
    except Exception as e:
        print(f"tensor_op_ok=False error={str(e)}")
    try:
        mean_val, dur_model = test_small_model(device)
        print(f"model_op_ok=True mean={mean_val:.4f} duration_s={dur_model:.4f}")
    except Exception as e:
        print(f"model_op_ok=False error={str(e)}")
    if device.type == "cuda":
        print("gpu_in_use=True")
    else:
        print("gpu_in_use=False")


if __name__ == "__main__":
    main()
