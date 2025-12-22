# Complete Guide: Installing SYCL Compiler (AdaptiveCpp) for NVIDIA GPUs

## Step 1: Install Miniconda

Miniconda is a minimal Python distribution with the conda package manager. We use it to install build tools without needing sudo.

Bash
'''
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
'''

What this does: Downloads and installs Miniconda to your home directory, then configures your shell to use it.

## Step 2: Create a Conda Environment

A conda environment is an isolated space where we install specific packages without affecting the rest of your system.

Bash
'''
conda create -n sycl python=3.11 -y
conda activate sycl
'''

What this does: Creates a new environment called sycl with Python 3.11, then activates it.

## Step 3: Install Build Dependencies

AdaptiveCpp needs LLVM/Clang compilers and other tools to build.

Bash
'''
conda install -c conda-forge clangxx cmake boost-cpp llvmdev lld clangdev -y
'''

What this does: Installs:

clangxx - The Clang C++ compiler
cmake - Build system generator
boost-cpp - C++ libraries AdaptiveCpp needs
llvmdev - LLVM development files
lld - LLVM's fast linker
clangdev - Clang header files
Step 4: Download AdaptiveCpp Source Code
AdaptiveCpp (formerly hipSYCL) is an open-source SYCL implementation that works with NVIDIA GPUs.

Bash
'''
cd ~
git clone --depth 1 https://github.com/AdaptiveCpp/AdaptiveCpp.git
'''

What this does: Downloads the latest AdaptiveCpp source code (shallow clone to save time/space).

## Step 5: Configure the Build

CMake configures how AdaptiveCpp will be built, telling it where to find CUDA and other dependencies.

Bash
'''
cd ~/AdaptiveCpp
mkdir build && cd build

cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/adaptivecpp \
  -DWITH_CUDA_BACKEND=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9 \
  -DCLANG_EXECUTABLE_PATH=$CONDA_PREFIX/bin/clang++ \
  -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm \
  -DCLANG_INCLUDE_PATH=$CONDA_PREFIX/lib/clang/21/include \
  -DACPP_LLD_PATH=$CONDA_PREFIX/bin/ld.lld \
  -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/clang \
  -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/clang++ \
  -DACPP_EXPERIMENTAL_LLVM=ON
'''

What each flag means:
Flag	Purpose
CMAKE_INSTALL_PREFIX	Where to install AdaptiveCpp (~/adaptivecpp)
WITH_CUDA_BACKEND=ON	Enable NVIDIA GPU support
CUDA_TOOLKIT_ROOT_DIR	Path to your CUDA installation
CLANG_EXECUTABLE_PATH	Path to the Clang compiler
LLVM_DIR	Path to LLVM cmake files
CLANG_INCLUDE_PATH	Path to Clang's internal headers
ACPP_LLD_PATH	Path to the LLD linker
ACPP_EXPERIMENTAL_LLVM=ON	Allow newer LLVM versions
Note: Adjust CUDA_TOOLKIT_ROOT_DIR to match your system's CUDA location.

## Step 6: Build and Install

Compile AdaptiveCpp (takes 5-15 minutes).

Bash
'''
make -j$(nproc)
make install
'''

What this does:

make -j$(nproc) - Compiles using all CPU cores
make install - Copies the built files to ~/adaptivecpp

## Step 7: Install Python Packages (for Jupyter Notebooks)

If you want to use SYCL from Jupyter notebooks:

Bash
'''
~/miniconda3/envs/sycl/bin/pip install tiktoken ollama requests ipykernel
~/miniconda3/envs/sycl/bin/python -m ipykernel install --user --name sycl --display-name "Python (sycl)"
'''

What this does: Installs Python packages and registers the sycl environment as a Jupyter kernel.

## Step 8: Set Up Environment Variables

These must be set every time before using SYCL.

Option A: Run manually each session:

Bash
'''
source ~/miniconda3/bin/activate sycl
export PATH=$HOME/adaptivecpp/bin:$PATH
export LD_LIBRARY_PATH=$HOME/adaptivecpp/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
'''

Option B: Add to ~/.bashrc for automatic setup:

Bash
'''
echo '
# SYCL/AdaptiveCpp setup
alias sycl-env="source ~/miniconda3/bin/activate sycl && export PATH=\$HOME/adaptivecpp/bin:\$PATH && export LD_LIBRARY_PATH=\$HOME/adaptivecpp/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/lib64:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
' >> ~/.bashrc
source ~/.bashrc
'''

Then just run sycl-env to activate everything.

## Step 9: Verify Installation

### Check compiler version
### Check if GPU is detected
Bash
'''
acpp --version
acpp-info
'''

Expected output from acpp-info:


Apply
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 1: CUDA
  Found device: NVIDIA RTX 5000 Ada Generation Laptop GPU   <-- Your GPU!

## How to Compile SYCL Code

Bash
'''
acpp --acpp-targets=cuda:sm_89 -O3 \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/math_libs/12.9/targets/x86_64-linux/include \
  my_program.cpp -o my_program
'''

Key flags:
Flag	Purpose
--acpp-targets=cuda:sm_89	Target NVIDIA GPU with compute capability 8.9 (RTX 5000 Ada)
-O3	Maximum optimization
-I...	Include paths for CUDA headers

## Troubleshooting
acpp-info shows no CUDA devices	- Run with LD_LIBRARY_PATH set correctly
CUDA version mismatch errors - Make sure CUDA_TOOLKIT_ROOT_DIR matches your driver's supported CUDA version
GPU shows 0% utilization - Code may be using CPU; ensure gpu_selector_v is used in SYCL code