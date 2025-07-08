#!/usr/bin/env python3
"""
华为昇腾NPU环境初始化脚本
用于检测NPU设备状态并进行基本配置
"""

import os
import sys
import torch

def check_npu_environment():
    """检查NPU环境是否正确配置"""
    print("=== 华为昇腾NPU环境检测 ===")
    
    # 检查torch_npu是否安装
    try:
        import torch_npu
        print("✓ torch_npu 已安装")
        print(f"  版本: {torch_npu.__version__}")
    except ImportError:
        print("✗ torch_npu 未安装")
        print("  请运行: pip install torch_npu")
        return False
    
    # 检查NPU是否可用
    if torch_npu.npu.is_available():
        print("✓ NPU 设备可用")
        device_count = torch_npu.npu.device_count()
        print(f"  检测到 {device_count} 个NPU设备")
        
        # 显示每个NPU设备信息
        for i in range(device_count):
            try:
                torch_npu.npu.set_device(i)
                device_name = torch_npu.npu.get_device_name(i)
                memory_total = torch_npu.npu.get_device_properties(i).total_memory
                print(f"  NPU {i}: {device_name}, 内存: {memory_total // (1024**3)} GB")
            except Exception as e:
                print(f"  NPU {i}: 无法获取详细信息 ({e})")
        
        return True
    else:
        print("✗ NPU 设备不可用")
        return False

def initialize_npu():
    """初始化NPU环境"""
    try:
        import torch_npu
        
        # 设置编译模式
        torch_npu.npu.set_compile_mode(jit_compile=False)
        print("✓ NPU 编译模式已设置")
        
        # 设置默认设备
        if torch_npu.npu.device_count() > 0:
            torch_npu.npu.set_device(0)
            print("✓ 默认NPU设备已设置为 npu:0")
        
        # 测试基本操作
        if torch_npu.npu.is_available():
            device = torch.device("npu:0")
            test_tensor = torch.randn(2, 2).to(device)
            result = test_tensor + test_tensor
            print("✓ NPU 基本运算测试通过")
            
            # 清理内存
            del test_tensor, result
            torch_npu.npu.empty_cache()
            print("✓ NPU 内存已清理")
        
        return True
        
    except Exception as e:
        print(f"✗ NPU 初始化失败: {e}")
        return False

def check_dependencies():
    """检查相关依赖包"""
    print("\n=== 依赖包检查 ===")
    
    required_packages = [
        'torch',
        'torch_npu', 
        'diffusers',
        'transformers',
        'accelerate',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            if package == 'torch':
                import torch
                print(f"✓ {package} {torch.__version__}")
            elif package == 'torch_npu':
                import torch_npu
                print(f"✓ {package} {torch_npu.__version__}")
            else:
                print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        return False
    
    return True

def main():
    """主函数"""
    print("华为昇腾NPU环境检测和初始化工具")
    print("=" * 50)
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查NPU环境
    npu_ok = check_npu_environment()
    
    if npu_ok:
        # 初始化NPU
        init_ok = initialize_npu()
        
        if init_ok:
            print("\n🎉 NPU环境配置完成！可以开始训练了。")
            print("\n使用建议:")
            print("- 运行前确保NPU驱动正常")
            print("- 监控NPU内存使用情况")
            print("- 如遇到问题，可以重新运行此脚本检查")
        else:
            print("\n❌ NPU初始化失败，请检查驱动和环境配置")
    else:
        print("\n❌ NPU环境不可用")
        if not deps_ok:
            print("请先安装缺少的依赖包")
        else:
            print("请检查NPU驱动是否正确安装")

if __name__ == "__main__":
    main() 