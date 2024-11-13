
<p align="center">
    <img src="https://github.com/user-attachments/assets/d8c62317-d82a-403a-b5da-7128c69225c7" alt="logo" width="400">
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Conda Installation](#conda-installation)
  - [Docker Installation](#docker-installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Output Structure](#output-structure)
  - [Viewing HTML Result Pages](#viewing-html-result-pages)
- [Models](#models)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)


## Introduction

**PEGASUS** is a sequence-based predictor of MD-derived information on protein flexibility. It generates embeddings using pre-trained models, predicts residue-wise real values of backbone fluctuation (RMSF), Phi & Psi dihedral angles standard deviation, and average Local Distance Difference Test (Mean LDDT) across the trajectory, and can optionally generate interactive result web pages for each protein sequence provided. PEGASUS accepts a FASTA (optionnaly aligned) file as input and allows users to specify the computation device (CPU or GPU).


## Features

- **Embedding Generation**: Generates protein embeddings using state-of-the-art pre-trained models:
  - ANKH Base (`ankh_base`)
  - ANKH Large (`ankh_large`)
  - ProtT5-XL UniRef50 (`prot_t5_xl_uniref50`)
  - ESM2 (`esm2_t36_3B_UR50D`)
- **Metric Predictions**: Predicts per-residue values for:
  - Root Mean Square Fluctuation (**RMSF**)
  - Standard Deviation of Phi Angles (**Std. Phi**)
  - Standard Deviation of Psi Angles (**Std. Psi**)
  - Mean Local Distance Difference Test (**Mean LDDT**)
- **Interactive Results**: Optionally generates interactive HTML pages for visualization, including a comprehensive results overview page with comparison functionality.
- **Aligned Sequence Support**: Supports aligned protein sequences as input, enabling the analysis of multiple sequence alignments.
- **HTTP Server**: Optionally starts an HTTP server to serve the result pages, making it easy to view results in a web browser.
- **Flexible Computation**: Supports CPU and GPU devices, with customizable device allocation for each model.
- **Reproducibility**: Allows setting a random seed for consistent results.
- **Docker Support**: Provides a Dockerfile for containerized execution.


## Installation

### Prerequisites

- **Operating System**: Linux or macOS (Windows not officially supported)
- **Python**: Version 3.8 or higher
- **CUDA**: For GPU support (optional)
- **Conda**: Package and environment management
- **Docker** (optional): For containerized execution

### 1. Clone the Repository

   ```bash
   git clone https://github.com/DSIMB/PEGASUS.git
   cd PEGASUS
   ```

### 2. Download Pegasus weights (1.3 Gb)

Download and extract Pegasus weights in the `models` directory.  

   ```bash
   aria2c -d ./models https://dsimb.inserm.fr/PEGASUS/models/pegasus_weights.tar.gz && tar -xzvf ~/models/pegasus_weights.tar.gz -C ./models
   ```

### 3. Conda Installation (not necessary if using Docker)

Create and Activate the Conda Environment

   ```bash
   conda env create -f pegasus.yml
   conda activate pegasus
   ```

### 3 (bis). Docker Installation

1) Install [NVIDIA Container Toolkit for GPU](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) support.
2) Setup running [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

3.1. **Build or download the Docker Image**

   ```bash
   docker pull dsimb/pegasus
   ```
   or download the bre-built latest docker image:
   ```bash
   docker build -t dsimb/pegasus .
   ```

3.2. **Run the Docker Container**

   ```bash
   docker run -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) --gpus all -v /path/to/input:/input -v /path/to/output:/output -v /path/to/models:/models dsimb/pegasus -i /input/sequences.fasta --output_dir /output --models_dir /models
   ```

   - Replace `/path/to/input`, `/path/to/output`, and `/path/to/models` with your local directories.
   - The `--gpus all` flag enables GPU support. Omit it if you wish to run on CPU.
   - The `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)` flags will generate the results owned by you, not root user when running docker with `sudo`.

> [!NOTE]  
> If cuda device is not detected in the docker container, try adding `--privileged` to the `docker run` command

## Usage

PEGASUS can be run via the command line. Below are the available command-line arguments and usage examples.

> [!WARNING]  
> Models are loaded on the selected device sequentially and purged, so the maximum memory required to run Pegasus corresponds to the largest model size, which is ESM2 with 11 Gb.

### Command-Line Arguments

```plaintext
usage: pegasus.py [-h] -i INPUT_FASTA [-d {cpu,gpu}] [-m MODELS_DIR] [-o OUTPUT_DIR]
                  [--model_device_map MODEL_DEVICE_MAP [MODEL_DEVICE_MAP ...]]
                  [-s SEED] [-t TOKS_PER_BATCH] [-g] [-k] [-a] [--serve] [--host HOST] [--port PORT]
```

| **Option**             | **Description**                                                                                                                                                 |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_fasta`          | **(Required)** Path to the input (multi)FASTA file containing protein sequences.                                                                                 |
| `default_device`       | Default computation device (`cpu` or `gpu`). Default is `cpu`.                                                                                                   |
| `models_dir`           | Directory containing pre-trained models. Default is the environment variable `MODELS_DIR` or `models`.                                                          |
| `output_dir`           | Directory to save output files. Default is the environment variable `OUTPUT_DIR` or `output`.                                                                   |
| `model_device_map`     | Specify device for each model in the format `model_name:device` (e.g., `prot_t5_xl_uniref50:gpu`). Accepted models: `ankh_base`, `ankh_large`, `prot_t5_xl_uniref50`, `esm2_t36_3B_UR50D`, `pegasus`. |
| `seed`                 | Random seed for reproducibility. Default is `42`.                                                                                                               |
| `toks_per_batch`       | Maximum tokens per batch to use during embedding generation. Default is `2048`.                                                                                 |
| `generate_html`        | Include this flag to generate result web pages for each protein as well as an overview page.                                                                    |
| `keep_embeddings`      | Keep the LLMs raw embeddings in `OUTPUT_EMBEDDINGS` directory after use. By default, the directory is deleted.                                                  |
| `aligned_fasta`        | Input protein sequences are aligned or not.                                                                                                                     |
| `serve`                | Start an HTTP server at the end to serve the result pages.                                                                                                      |
| `host`                 | Hostname to use when serving the result pages. Default is `"localhost"`.                                                                                        |
| `port`                 | Port to use when serving the result pages. Default is `8000`.                                                                                                   |


### Examples

1. **Basic Usage**

   Generate embeddings and predictions using default settings:

   ```bash
   python pegasus.py -i sequences.fasta
   ```

2. **Using GPU for All Computations**

   ```bash
   python pegasus.py -i sequences.fasta -d gpu
   ```

3. **Specify Devices for Specific Models**

   ```bash
   python pegasus.py -i sequences.fasta -d gpu --model_device_map esm2_t36_3B_UR50D:cpu prot_t5_xl_uniref50:cpu
   ```

4. **Specify that input protein sequences (multifasta) are aligned**

   ```bash
   python pegasus.py -i sequences.fasta --aligned_fasta
   ```

5. **Generate Interactive HTML Result Pages**

   ```bash
   python pegasus.py -i sequences.fasta --generate_html
   ```

6. **Full Command with Custom Output and Models Directory**

   ```bash
   python pegasus.py -i sequences.fasta --output_dir /path/to/output --models_dir /path/to/models --generate_html
   ```

7. **Full Command with Custom Output and Models Directory and serve the result pages**

   ```bash
   python pegasus.py -i sequences.fasta --output_dir /path/to/output --models_dir /path/to/models --generate_html --serve
   ```



## Output Structure

After running PEGASUS, a unique output directory (e.g. `output/0d6d0268/`) will contain:

- **Embeddings**: Stored in `embeddings/` subdirectory, organized by model name, if requested to be kept with command line argument `--keep-embeddings` (deleted by default).
- **Predictions**: Stored in `predictions/` subdirectory, includes the per-protein TSV files with predictions.
- **Result Pages**: If `--generate_html` is used, interactive HTML pages are stored in `result_pages/`.
- **`id_mapping.tsv`**: Two columns TSV file containing a mapping of the unique `Generated_ID` (P{1..n}) generated by Pegasus and the `Original_ID` fasta header of each input protein.


## View html result pages

If `--generate_html` is used, interactive HTML pages are stored in `result_pages/`.

1.	Using the Built-in HTTP Server

   If you used the `--serve` flag, the result pages are automatically served via an HTTP server. By default, you can access the results at http://localhost:8000/results_overview.html.

2.	Manual HTTP Server

   ```bash
   cd output/{job_id}/result_pages
   python -m http.server
   Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
   ```

Access the pages on your browser at this URL: http://0.0.0.0:8000

## Models

PEGASUS utilizes several pre-trained models for embedding generation:

- **ANKH Base and Large** [paper](https://arxiv.org/pdf/2301.06568)
- **ProtT5-XL UniRef50** [paper](https://ieeexplore.ieee.org/document/9477085)
- **ESM2 (esm2_t36_3B_UR50D)** [paper](https://www.science.org/doi/abs/10.1126/science.ade2574)

**Note**: The models are automatically downloaded and saved to the `--models_dir` directory upon first use.


## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.


## Citation

If you use PEGASUS in your research, please cite:


## Contact

For questions, feedback, or issues, please open an issue on the [GitHub repository](https://github.com/DSIMB/PEGASUS/issues)
