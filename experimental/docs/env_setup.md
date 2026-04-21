# Environment Setup

1. We use conda to install required packages and recommend creating a new environment for the codebase:
    ```bash
        conda create -n equiformer_v3 python=3.11 -c conda-forge
        conda activate equiformer_v3
    ```

2. Clone the repository:
    ```bash
        git clone https://github.com/atomicarchitects/equiformer_v3.git
        cd equiformer_v3
    ```

3. Here we use CUDA 12.8 and install python packages as follows:
    ```bash
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
        pip install torch_geometric

        pip install -r experimental/env/conda_requirements.txt
    ```

    To use CPU wheels (macOS / no CUDA):
    ```bash
        pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cpu.html
        pip install torch_geometric

        grep -v "triton" experimental/env/conda_requirements.txt | pip install -r /dev/stdin
    ```
    > Note: `triton` in `conda_requirements.txt` is Linux-only and unavailable on macOS — the `grep -v` above skips it.

4. We install the `fairchem` package inside this repository by running:
    ```bash
        pip install -e packages/fairchem-core
    ```

5. We install Matbench Discovery package by running:
    ```bash
        git clone https://github.com/janosh/matbench-discovery.git
        cd matbench-discovery
        git checkout 375a8d6
        pip install -e .
    ```
