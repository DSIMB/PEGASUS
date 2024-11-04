# PEGASUS: Protein Embedding and Generative Analysis Script Using Sequences

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
- [Models](#models)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)


## Introduction

**PEGASUS** (Protein Embedding and Generative Analysis Script Using Sequences) is a comprehensive tool designed for protein sequence analysis. It generates embeddings using pre-trained models, predicts various structural and functional metrics, and can optionally generate interactive result web pages for each protein sequence provided. PEGASUS accepts a FASTA file as input and allows users to specify the computation device (CPU or GPU).


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
- **Interactive Results**: Optionally generates interactive HTML pages for visualization for the same user experience as with the webserver.
- **Flexible Computation**: Supports CPU and GPU devices, with customizable device allocation for each model.
- **Reproducibility**: Allows setting a random seed for consistent results.
- **Docker Support**: Provides a Dockerfile for containerized execution.


## Installation

### Prerequisites

- **Operating System**: Linux or macOS (Windows not officially supported)
- **Python**: Version 3.8 or higher
- **CUDA**: For GPU support (optional)
- **Conda**: Package and environment management
- **Docker**: For containerized execution (optional)

### Conda Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DSIMB/PEGASUS.git
   cd PEGASUS
   ```

2. **Create and Activate the Conda Environment**

   ```bash
   conda env create -f pegasus.yml
   conda activate pegasus
   ```


### Docker Installation

1. **Build or download the Docker Image**

   ```bash
   docker pull dsimb/pegasus
   ```

   ```bash
   docker build -t dsimb/pegasus .
   ```

2. **Run the Docker Container**

   ```bash
   docker run --gpus all -v /path/to/input:/input -v /path/to/output:/output -v /path/to/models:/models dsimb/pegasus -i ./input/sequences.fasta --output_dir ./output --models_dir ./models
   ```

   - Replace `/path/to/input`, `/path/to/output`, and `/path/to/models` with your local directories.
   - The `--gpus all` flag enables GPU support. Omit it if you wish to run on CPU.


## Usage

PEGASUS can be run via the command line. Below are the available command-line arguments and usage examples.

### Command-Line Arguments

```plaintext
usage: pegasus.py [-h] -i INPUT_FASTA [-d {cpu,gpu}] [--models_dir MODELS_DIR]
                  [--output_dir OUTPUT_DIR]
                  [--model_device_map MODEL_DEVICE_MAP [MODEL_DEVICE_MAP ...]]
                  [--seed SEED] [--generate_html]
```

- `-i`, `--input_fasta`: **(Required)** Path to the input (multi)FASTA file containing protein sequences.
- `-d`, `--default_device`: Default computation device (`cpu` or `gpu`). Default is `cpu`.
- `--models_dir`: Directory containing pre-trained models. Default is the environment variable `MODELS_DIR` or `models`.
- `--output_dir`: Directory to save output files. Default is the environment variable `OUTPUT_DIR` or `output`.
- `--model_device_map`: Specify device for each model in the format `model_name:device` (e.g., `prot_t5_xl_uniref50:gpu`). Accepted models are `ankh_base`, `ankh_large`, `prot_t5_xl_uniref50`, `esm2_t36_3B_UR50D`, `pegasus`.
- `--seed`: Random seed for reproducibility. Default is `42`.
- `--generate_html`: Include this flag to generate result web pages for each protein as well as an overview page.

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
   python pegasus.py -i sequences.fasta --model_device_map ankh_base:cpu prot_t5_xl_uniref50:gpu
   ```

4. **Generate Interactive HTML Result Pages**

   ```bash
   python pegasus.py -i sequences.fasta --generate_html
   ```

5. **Full Command with Custom Output and Models Directory**

   ```bash
   python pegasus.py -i sequences.fasta --output_dir /path/to/output --models_dir /path/to/models --generate_html
   ```


## Output Structure

After running PEGASUS, a unique output directory (e.g. `output/0d6d0268/`) will contain:

- **Embeddings**: Stored in `embeddings/` subdirectory, organized by model name.
- **Predictions**: Stored in `predictions/` subdirectory, includes:
  - `results.pkl`: Pickle file containing prediction results.
  - Per-protein TSV files with predictions.
- **Result Pages**: If `--generate_html` is used, interactive HTML pages are stored in `result_pages/`.
- **`id_mapping.tsv`**: Two columns TSV file containing a mapping of the unique `Generated_ID` generated by Pegasus and the `Original_ID` fasta header of each input protein.



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

For questions, feedback, or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/pegasus/issues)