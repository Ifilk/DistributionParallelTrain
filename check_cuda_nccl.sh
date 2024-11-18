#!/bin/bash

# 设置代理（根据需要修改）
export HTTP_PROXY=http://your.proxy.address:port
export HTTPS_PROXY=http://your.proxy.address:port

# 检查命令是否可用的函数
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查并打印CUDA版本
check_cuda() {
    echo "Checking CUDA installation..."
    if command_exists nvcc; then
        echo "CUDA is installed."
        nvcc_version=$(nvcc --version | grep -oP "release \K[\d.]+")
        echo "CUDA Version: $nvcc_version"
    else
        echo "CUDA is not installed."
        install_cuda
    fi
}

# 安装CUDA函数
install_cuda() {
    echo "Installing CUDA..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # 检查系统发行版
        if command_exists apt; then
            echo "Detected Ubuntu-based system. Installing CUDA from NVIDIA repository..."
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
            sudo dpkg -i cuda-keyring_1.0-1_all.deb
            sudo apt-get update
            sudo apt-get install -y cuda
        elif command_exists yum; then
            echo "Detected RHEL-based system. Installing CUDA from NVIDIA repository..."
            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
            sudo dnf clean all
            sudo dnf -y install cuda
        else
            echo "Unsupported package manager. Please install CUDA manually."
            exit 1
        fi
    else
        echo "Non-Linux systems are not supported by this script."
        exit 1
    fi
}

# 检查并打印NCCL状态
check_nccl() {
    echo "Checking NCCL installation..."
    if command_exists nvidia-smi; then
        echo "NVIDIA driver detected."
        if dpkg -l | grep -q libnccl2; then
            echo "NCCL is installed."
        else
            echo "NCCL is not installed."
            install_nccl
        fi
    else
        echo "NVIDIA driver is not detected. Please install the driver first."
        exit 1
    fi
}

# 安装NCCL函数
install_nccl() {
    echo "Installing NCCL..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt; then
            echo "Installing NCCL for Ubuntu..."
            sudo apt-get install -y libnccl2 libnccl-dev
        elif command_exists yum; then
            echo "Installing NCCL for RHEL-based system..."
            sudo dnf install -y libnccl libnccl-devel
        else
            echo "Unsupported package manager. Please install NCCL manually."
            exit 1
        fi
    else
        echo "Non-Linux systems are not supported by this script."
        exit 1
    fi
}

# 检查代理设置
check_proxy() {
    echo "Checking proxy settings..."
    if [[ -z "$HTTP_PROXY" || -z "$HTTPS_PROXY" ]]; then
        echo "Proxy is not set. You may encounter network issues."
        echo "Set HTTP_PROXY and HTTPS_PROXY variables in the script if needed."
    else
        echo "Proxy is set."
    fi
}

# 主函数
main() {
    check_proxy
    check_cuda
    check_nccl
    echo "All checks passed. System is ready for distributed training."
}

# 执行主函数
main
