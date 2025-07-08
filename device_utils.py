"""
设备兼容性工具模块
提供CUDA和NPU的统一接口
"""

import torch

# 检查是否有torch_npu
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

def get_device_type():
    """获取当前设备类型"""
    if HAS_NPU and torch_npu.npu.is_available():
        return "npu"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def device_count():
    """获取设备数量"""
    device_type = get_device_type()
    if device_type == "npu":
        return torch_npu.npu.device_count()
    elif device_type == "cuda":
        return torch.cuda.device_count()
    else:
        return 1

def empty_cache():
    """清空设备缓存"""
    device_type = get_device_type()
    if device_type == "npu":
        torch_npu.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()

def create_event(enable_timing=True):
    """创建设备事件"""
    device_type = get_device_type()
    if device_type == "npu":
        return torch_npu.npu.Event(enable_timing=enable_timing)
    elif device_type == "cuda":
        return torch.cuda.Event(enable_timing=enable_timing)
    else:
        # CPU情况下返回一个简单的时间记录器
        return CPUEvent(enable_timing=enable_timing)

def synchronize(device=None):
    """同步设备"""
    device_type = get_device_type()
    if device_type == "npu":
        torch_npu.npu.synchronize(device)
    elif device_type == "cuda":
        torch.cuda.synchronize(device)

class CPUEvent:
    """CPU事件模拟器"""
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing
        self.start_time = None
        self.end_time = None
        
    def record(self):
        if self.enable_timing:
            import time
            if self.start_time is None:
                self.start_time = time.time()
            else:
                self.end_time = time.time()
    
    def elapsed_time(self, end_event):
        if self.enable_timing and self.start_time and end_event.end_time:
            return (end_event.end_time - self.start_time) * 1000  # 转换为毫秒
        return 0.0 