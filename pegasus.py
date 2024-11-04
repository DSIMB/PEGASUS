#!/usr/bin/env python3
"""
PEGASUS: Protein Embedding and Generative Analysis Script Using Sequences

This script generates embeddings for protein sequences using pre-trained models,
makes predictions for various metrics, and optionally generates result web pages
for each protein. It accepts a FASTA file as input and allows the user to specify
the computation device (CPU or GPU).
"""

import argparse
import logging
import os
import random
import sys
import time
import warnings
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datetime import datetime
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

import esm
import ankh
import result_page_generator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Adjust transformers logging level
transformers.logging.set_verbosity_error()

# Accepted models for device mapping
ACCEPTED_MODELS = ['ankh_base', 'ankh_large', 'prot_t5_xl_uniref50', 'esm2_t36_3B_UR50D', 'pegasus']


def set_seed(seed=42):
    """
    Fix the seeds for reproducible runs during training.

    Args:
        seed (int): Seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings and predictions for protein sequences."
    )
    parser.add_argument(
        "-i", "--input_fasta",
        type=str,
        required=True,
        help="Path to the input FASTA file containing protein sequences."
    )
    parser.add_argument(
        "-d", "--default_device",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="Default computation device to use ('cpu' or 'gpu') for models not specified in --model_device_map. Default is 'cpu'."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=os.environ.get('MODELS_DIR', 'models'),
        help="Path to the directory containing pre-trained models."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get('OUTPUT_DIR', 'output'),
        help="Directory to save output files."
    )
    parser.add_argument(
        "--model_device_map",
        type=str,
        nargs='+',
        help=f"Specify device for each model in the format model_name:device (e.g., prot_t5_xl_uniref50:cpu). Accepted models are: {', '.join(ACCEPTED_MODELS)}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=2048,
        help="Tokens per batch to use during embedding generation. Default is 3074."
    )
    parser.add_argument(
        "--generate_html",
        action='store_true',
        help="Generate result web pages for each protein."
    )
    return parser.parse_args()


def process_model_device_map(model_device_list):
    """
    Process the model-device mapping from command-line arguments.

    Args:
        model_device_list (list): List of strings in the format model_name:device.

    Returns:
        dict: Dictionary mapping model names to devices.
    """
    model_device_map = {}
    for item in model_device_list:
        try:
            model_name, device_name = item.split(':')
            if model_name not in ACCEPTED_MODELS:
                logging.error(f"Invalid model '{model_name}'. Accepted models are: {', '.join(ACCEPTED_MODELS)}.")
                sys.exit(1)
            if device_name not in ['cpu', 'gpu']:
                logging.error(f"Invalid device '{device_name}' for model '{model_name}'. Must be 'cpu' or 'gpu'.")
                sys.exit(1)
            model_device_map[model_name] = device_name
        except ValueError:
            logging.error(f"Invalid format for --model_device_map item '{item}'. Expected format is model_name:device.")
            sys.exit(1)
    return model_device_map


def validate_fasta(fasta_path):
    """
    Validate the FASTA file and extract sequences and labels.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        tuple: A tuple containing a list of generated labels, a list of sequences, and a mapping dictionary.
    """
    if not os.path.isfile(fasta_path):
        logging.error(f"The file '{fasta_path}' does not exist.")
        sys.exit(1)

    allowed_letters = set("ACDEFGHIKLMNPQRSTVWY")
    sequences = []
    labels = []
    id_mapping = {}

    try:
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            seq = str(record.seq).upper()
            if not set(seq).issubset(allowed_letters):
                logging.error(f"Invalid characters found in sequence '{record.id}'. Allowed amino acids are: {allowed_letters}")
                sys.exit(1)
            sequences.append(seq)
            generated_id = f"P{i}"
            labels.append(generated_id)
            id_mapping[generated_id] = record.description
    except Exception as e:
        logging.error(f"Error parsing FASTA file: {e}")
        sys.exit(1)

    if not sequences:
        logging.error("No valid sequences found in the FASTA file.")
        sys.exit(1)

    return labels, sequences, id_mapping


class FastaBatchedDataset:
    """
    A unified dataset class that can handle both standard and T5-specific batching.
    """
    def __init__(self, sequence_labels, sequence_strs, sequence_toks=None, is_t5=False):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.is_t5 = is_t5
        self.sequence_toks = list(sequence_toks) if is_t5 else None

    @classmethod
    def from_list(cls, sequences, labels=None, is_t5=False):
        sequence_labels = []
        sequence_strs = []
        sequence_toks = []
        if labels is None:
            labels = [f"sequence_{i}" for i in range(len(sequences))]
        for label, seq in zip(labels, sequences):
            sequence_labels.append(label)
            sequence_strs.append(seq)
            if is_t5:
                sequence_toks.append(" ".join(list(seq)))
        return cls(sequence_labels, sequence_strs, sequence_toks if is_t5 else None, is_t5=is_t5)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        if self.is_t5:
            return self.sequence_labels[idx], self.sequence_strs[idx], self.sequence_toks[idx]
        else:
            return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


def load_model_and_tokenizer(name: str, device, models_dir: str):
    """
    Load the specified model and tokenizer.

    Args:
        name (str): Name of the model.
        device (torch.device): Computation device.
        models_dir (str): Directory containing the pre-trained models.

    Returns:
        tuple: Model, tokenizer.

    Raises:
        ValueError: If the model name is unknown.
    """
    if name in ['ankh_base', 'ankh_large']:
        model_path = os.path.join(models_dir, f'{name}.pt')
        tokenizer_path = os.path.join(models_dir, f'{name}_tokenizer.pt')
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=device).eval()
            tokenizer = torch.load(tokenizer_path, map_location=device)
        else:
            if name == 'ankh_base':
                model, tokenizer = ankh.load_base_model()
            else:
                model, tokenizer = ankh.load_large_model()
            model = model.to(device)
            # Save the model and tokenizer
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model, model_path)
            torch.save(tokenizer, tokenizer_path)
    elif name == 'prot_t5_xl_uniref50':
        model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50', cache_dir=models_dir).to(device).eval()
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False, cache_dir=models_dir)
    elif name == 'esm2_t36_3B_UR50D':
        model_path = os.path.join(models_dir, f'{name}.pt')
        tokenizer_path = os.path.join(models_dir, f'{name}_tokenizer.pt')
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            model = torch.load(model_path)
            alphabet = torch.load(tokenizer_path)
            tokenizer = alphabet.get_batch_converter()
            model = model.to(device).eval()
        else:
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            model = model.to(device).eval()
            tokenizer = alphabet.get_batch_converter()
            # Save the model and tokenizer
            os.makedirs(models_dir, exist_ok=True)
            torch.save(model, model_path)
            torch.save(alphabet, tokenizer_path)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return model, tokenizer


def embed_ankh(seqs, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using Ankh models.

    Args:
        seqs (list): List of sequences.
        tokenizer: Tokenizer for the model.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        list: List of embeddings.
    """
    results = []
    dataset = FastaBatchedDataset.from_list(seqs, is_t5=False)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches)
    with torch.no_grad():
        for labels, strs in tqdm(data_loader, desc="Generating embeddings with Ankh models"):
            sequences = [list(seq) for seq in strs]
            outputs = tokenizer.batch_encode_plus(
                sequences,
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt"
            )
            input_ids = outputs['input_ids'].to(device)
            attention_mask = outputs['attention_mask'].to(device)
            embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            for j, emb in enumerate(embeddings):
                seq_len = (attention_mask[j] == 1).sum() - 1
                clean_emb = emb[1:seq_len+1].cpu()
                results.append(clean_emb)
    return results


def embed_prot_t5(seqs, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using ProtT5 model.

    Args:
        seqs (list): List of sequences.
        tokenizer: Tokenizer for the model.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        list: List of embeddings.
    """
    results = []
    dataset = FastaBatchedDataset.from_list(seqs, is_t5=True)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches)
    with torch.no_grad():
        for labels, strs, toks in tqdm(data_loader, desc="Generating embeddings with ProtT5"):
            ids = tokenizer.batch_encode_plus(toks, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            embedding = encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = embedding.last_hidden_state.cpu()
            for i in range(len(embeddings)):
                seq_len = (attention_mask[i] == 1).sum()
                seq_emb = embeddings[i][:seq_len-1]
                results.append(seq_emb)
    return results


def embed_esm2(seqs, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using ESM2 model.

    Args:
        seqs (list): List of sequences.
        tokenizer: Batch converter obtained from the alphabet.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        list: List of embeddings.
    """
    results = []
    dataset = FastaBatchedDataset.from_list(seqs, is_t5=False)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    batch_converter = tokenizer  # tokenizer is actually the batch_converter
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=batches)
    with torch.no_grad():
        for labels, strs, tokens in tqdm(data_loader, desc="Generating embeddings with ESM2"):
            tokens = tokens.to(device)
            outputs = encoder(tokens, repr_layers=[33], return_contacts=False)
            representations = outputs['representations'][33].cpu()
            for j, seq_emb in enumerate(representations):
                seq_len = len(strs[j])
                seq_emb = seq_emb[1:seq_len+1]
                results.append(seq_emb)
    return results


def get_embeddings(seqs, labels, models_list, toks_per_batch, output_embeddings_dir, default_device, model_device_map, models_dir):
    """
    Generate embeddings for the given sequences using specified models.

    Args:
        seqs (list): List of sequences.
        labels (list): List of sequence labels.
        models_list (list): List of model names.
        toks_per_batch (int): Tokens per batch.
        output_embeddings_dir (str): Directory to save embeddings.
        default_device (str): Default computation device.
        model_device_map (dict): Mapping of model names to devices.
        models_dir (str): Directory containing the pre-trained models.
    """
    # Load models
    loaded_models = {}
    for model_name in models_list:
        device_choice = model_device_map.get(model_name, default_device)
        device = torch.device("cuda" if device_choice == 'gpu' and torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model and tokenizer for {model_name} on device {device}...")
        model, tokenizer = load_model_and_tokenizer(model_name, device, models_dir)
        loaded_models[model_name] = (model, tokenizer, device)

    # For each model, generate embeddings
    for model_name in models_list:
        model, tokenizer, device = loaded_models[model_name]
        logging.info(f"Generating embeddings using {model_name} on device {device}...")
        if model_name in ['ankh_base', 'ankh_large']:
            embeddings = embed_ankh(seqs, tokenizer, model, device, toks_per_batch)
        elif model_name == 'prot_t5_xl_uniref50':
            embeddings = embed_prot_t5(seqs, tokenizer, model, device, toks_per_batch)
        elif model_name == 'esm2_t36_3B_UR50D':
            embeddings = embed_esm2(seqs, tokenizer, model, device, toks_per_batch)
        else:
            raise ValueError(f"Unknown model {model_name}")

        # Save embeddings
        output_dir = os.path.join(output_embeddings_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving embeddings for {model_name}...")
        for idx, emb in enumerate(embeddings):
            label = labels[idx]
            result = {
                "label": label,
                "embedding": emb
            }
            output_file = os.path.join(output_dir, f"{label}.pt")
            torch.save(result, output_file)


class CONV_3L(nn.Module):
    """
    Convolutional Neural Network with 3 layers.
    """
    def __init__(self, embd_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=embd_dim, out_channels=128, kernel_size=15, padding="same")
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=5, padding="same")
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, padding="same")

        # Batch normalization
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(16)

        # Define the activation function
        self.tanh = nn.Tanh()

        # Define dropout
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.tanh(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.tanh(out)
        out = self.conv3(out)
        out = out.permute(0, 2, 1)
        return out


class CONV_2(nn.Module):
    """
    Convolutional Neural Network with 2 layers.
    """
    def __init__(self, embd_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=embd_dim, out_channels=int(embd_dim / 4), kernel_size=15, padding="same")
        self.conv2 = nn.Conv1d(in_channels=int(embd_dim / 4), out_channels=1, kernel_size=5, padding="same")

        # Batch normalization and maxpool
        self.batchnorm1 = nn.BatchNorm1d(int(embd_dim / 4))

        # Define the activation function
        self.tanh = nn.Tanh()

        # Define dropout
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.tanh(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        return out


def read_and_sample_pids(folder_path):
    """
    Read all IDs in a folder and return a list.

    Args:
        folder_path (str): Path to the folder where embeddings are.

    Returns:
        list: Available IDs in that embeddings folder.
    """
    pdbs = [f[:-3] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return pdbs


def predict_protein_metrics(X, model, length, default_device, model_device_map):
    """
    Generate predictions for a single protein embedding.

    Args:
        X (torch.Tensor): Embedding matrix.
        model (nn.Module): Neural network model.
        length (int): Length of the protein sequence.
        default_device (str): Default computation device.
        model_device_map (dict): Mapping of model names to devices.

    Returns:
        list: Predicted metrics for the protein embedding.
    """
    prot_length = length
    device_choice = model_device_map.get("pegasus", default_device)
    device = torch.device("cuda" if device_choice == 'gpu' and torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        X = X[None, :, :].to(device)
        pred = model(X)
        pred = torch.squeeze(pred, axis=2)
        pred_prot = pred.detach().cpu().numpy()
        pred_prot = pred_prot[0][:prot_length]
        pred_prot = [0 if i < 0 else i for i in pred_prot]
        return pred_prot


def predict_metrics_for_proteins(protein_ids, available_X, available_metrics, model_path, default_device, model_device_map):
    """
    Load data and make predictions for a list of proteins.

    Args:
        protein_ids (iterable): Names of protein IDs.
        available_X (list): Paths to the embeddings.
        available_metrics (list): Paths to the folders with model data for each metric.
        model_path (list): Names of each model.
        default_device (str): Default computation device.
        model_device_map (dict): Mapping of model names to devices.

    Returns:
        tuple: Dictionary of results and list of protein IDs.
    """
    results_dict = {}
    processed_protein_ids = []
    model_cache = {}

    device_choice = model_device_map.get("pegasus", default_device)
    device = torch.device("cuda" if device_choice == 'gpu' and torch.cuda.is_available() else "cpu")
    logging.info(f"Predict metrics on device {device}...")

    for pid in tqdm(protein_ids, desc='Predicting metrics'):
        pdb_id = pid

        # Load embedding to get length
        embedding_path = os.path.join(available_X[0], f"{pdb_id}.pt")
        if not os.path.exists(embedding_path):
            continue  # Skip if embedding does not exist

        embedding_data = torch.load(embedding_path)
        length = embedding_data["embedding"].shape[0]

        # Verify that sequence is not too long
        if length > SEQ_LENGTH_THRESHOLD:
            continue

        # Load all embeddings for pid
        embeddings = {}
        embd_dims = {}
        for embedding in available_X:
            embedding_file = os.path.join(embedding, f"{pdb_id}.pt")
            if not os.path.exists(embedding_file):
                continue  # Skip if embedding file does not exist
            X = torch.load(embedding_file)["embedding"]
            embd_dim = X.shape[1]
            embd_dims[embedding] = embd_dim

            X_padded = np.zeros((SEQ_LENGTH_THRESHOLD, embd_dim))
            X_padded[:len(X), :] = X
            X_padded = torch.from_numpy(X_padded.astype(np.float32)).to(device)
            embeddings[embedding] = X_padded

        if not embeddings:
            continue  # Skip if no embeddings loaded

        # Initialize metrics
        metric_dict = {'RMSF': None, 'PHI': None, 'PSI': None, 'LDDT': None}

        # To save each individual embedding prediction
        all_predictions = {}

        for metric_path, metric_name in zip(available_metrics, metric_dict):
            mean_of_predictions = np.zeros(length, dtype='float64')
            n_embeddings_used = 0

            for embedding, model_name in zip(available_X, model_path):
                if embedding not in embeddings:
                    continue  # Skip if embedding not loaded
                X_padded = embeddings[embedding]
                embd_dim = embd_dims[embedding]

                # Load or get model from cache
                key = (metric_path, model_name, embd_dim)
                if key not in model_cache:
                    # Load model
                    if metric_path == RMSF_MODEL_PATH or metric_path == LDDT_MODEL_PATH:
                        model = CONV_3L(embd_dim)
                    else:
                        model = CONV_2(embd_dim)
                    path = os.path.join(metric_path, model_name)
                    if not os.path.exists(path):
                        continue  # Skip if model file does not exist
                    states = torch.load(path, map_location=device)['model_state_dict']
                    model.load_state_dict(states, strict=True)
                    model.to(device)
                    model.eval()
                    model_cache[key] = model
                else:
                    model = model_cache[key]

                # Generate a prediction
                predictions = predict_protein_metrics(X_padded, model, length, default_device, model_device_map)

                # Append prediction to dictionary
                model_short_name = model_name.split('_CV1.pth')[0]
                all_predictions[f'{metric_name}_{model_short_name}'] = predictions

                # Add to mean vector
                mean_of_predictions += np.array(predictions)
                n_embeddings_used += 1

            # Calculate the mean of predictions
            if n_embeddings_used > 0:
                mean_of_predictions /= n_embeddings_used
            else:
                mean_of_predictions = np.zeros(length, dtype='float64')
            metric_dict[metric_name] = mean_of_predictions

        # Save mean results into the first dictionary
        processed_protein_ids.append(pdb_id)
        results_dict[pdb_id] = metric_dict

        # Save per embedding data into a tsv file
        df = pd.DataFrame(all_predictions)
        df.to_csv(os.path.join(RESULT_PATH, f'{pdb_id}_all_predictions.tsv'), sep='\t')

    return results_dict, processed_protein_ids

def format_duration(seconds):
    """
    Format the duration into minutes and seconds.
    """
    if seconds >= 60:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes} min {seconds} sec" if seconds else f"{minutes} min"
    else:
        return f"{seconds} sec"


def generate_results_overview_page(labels, seqs, id_mapping, results_dict, unique_output_dir, start_time, run_id):
    """
    Generate the results overview page with comparison functionality.
    """
    # Compute job duration
    end_time = time.time()
    job_duration = int(end_time - start_time)
    job_duration_str = format_duration(job_duration)

    # Prepare data for the overview page
    headers = [id_mapping[label] for label in labels]
    protein_ids = labels
    sequences = seqs
    date = datetime.today().strftime('%Y-%m-%d')

    # Path to save the results overview page
    output_html_dir = os.path.join(unique_output_dir, 'result_pages')
    os.makedirs(output_html_dir, exist_ok=True)

    # Call the function to write the results overview page
    result_page_generator.write_results_overview_page(
        job_id=run_id,
        job_duration=job_duration_str,
        date=date,
        headers=headers,
        sequences=sequences,
        protein_ids=protein_ids,
        results_dict=results_dict,
        output_dir=output_html_dir
    )


def main():
    """
    Main function to run the PEGASUS pipeline.
    """
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Generate a unique run ID
    run_id = str(uuid.uuid4())[:8]  # Use the first 8 characters for brevity

    # Create a unique output directory for this run
    unique_output_dir = os.path.join(args.output_dir, f"{run_id}")
    os.makedirs(unique_output_dir, exist_ok=True)

    # Set seed
    set_seed(args.seed)

    # Validate and parse FASTA file
    labels, seqs, id_mapping = validate_fasta(args.input_fasta)

    # Save the id mapping to a file
    mapping_file = os.path.join(unique_output_dir, 'id_mapping.tsv')
    with open(mapping_file, 'w') as f:
        f.write("Generated_ID\tOriginal_ID\n")
        for gen_id, orig_id in id_mapping.items():
            f.write(f"{gen_id}\t{orig_id}\n")

    # Process model-device mapping
    model_device_map = {}
    if args.model_device_map:
        model_device_map = process_model_device_map(args.model_device_map)
    default_device = args.default_device

    # Constants and configurations
    global SEQ_LENGTH_THRESHOLD
    SEQ_LENGTH_THRESHOLD = args.toks_per_batch
    toks_per_batch = args.toks_per_batch

    # Output directories
    OUTPUT_EMBEDDINGS = os.path.join(unique_output_dir, 'embeddings')
    os.makedirs(OUTPUT_EMBEDDINGS, exist_ok=True)

    # Models to generate embeddings
    models_list = ['prot_t5_xl_uniref50', 'esm2_t36_3B_UR50D', 'ankh_base', 'ankh_large']

    # Generate embeddings
    logging.info("Starting embedding generation...")
    get_embeddings(seqs, labels, models_list, toks_per_batch, OUTPUT_EMBEDDINGS, default_device, model_device_map, args.models_dir)
    logging.info("Embedding generation completed.")

    # Second step - read embeddings and generate predictions
    X_ankhL = os.path.join(OUTPUT_EMBEDDINGS, 'ankh_large')
    X_ankhB = os.path.join(OUTPUT_EMBEDDINGS, 'ankh_base')
    X_esm36 = os.path.join(OUTPUT_EMBEDDINGS, 'esm2_t36_3B_UR50D')
    X_t5 = os.path.join(OUTPUT_EMBEDDINGS, 'prot_t5_xl_uniref50')

    # Path to models for each metric
    global RMSF_MODEL_PATH, PHI_MODEL_PATH, PSI_MODEL_PATH, LDDT_MODEL_PATH
    RMSF_MODEL_PATH = os.path.join(args.models_dir, 'pegasus/weights/RMSF')
    PHI_MODEL_PATH = os.path.join(args.models_dir, 'pegasus/weights/PHI')
    PSI_MODEL_PATH = os.path.join(args.models_dir, 'pegasus/weights/PSI')
    LDDT_MODEL_PATH = os.path.join(args.models_dir, 'pegasus/weights/LDDT')

    # Model IDs
    ankhL = 'ankh_large_CV1.pth'
    ankhB = 'ankh_base_CV1.pth'
    esm36 = 'esm2_t36_3B_UR50D_CV1.pth'
    t5 = 'prot_t5_xl_uniref50_CV1.pth'

    # Set embeddings used to train the models
    X_paths = [X_ankhL, X_ankhB, X_esm36, X_t5]
    models_names = [ankhL, ankhB, esm36, t5]
    metrics = [RMSF_MODEL_PATH, PHI_MODEL_PATH, PSI_MODEL_PATH, LDDT_MODEL_PATH]

    # Path to save results
    global RESULT_PATH
    RESULT_PATH = os.path.join(unique_output_dir, 'predictions')
    os.makedirs(RESULT_PATH, exist_ok=True)

    # Get list of protein IDs
    protein_ids = read_and_sample_pids(X_ankhL)

    # Fit the model and make predictions
    logging.info("Starting predictions...")
    results_dict, processed_protein_ids = predict_metrics_for_proteins(
        protein_ids, X_paths, metrics, models_names, default_device, model_device_map)
    logging.info("Predictions completed.")

    # Save results
    import pickle
    with open(os.path.join(RESULT_PATH, 'results.pkl'), 'wb') as fp:
        pickle.dump(results_dict, fp)

    with open(os.path.join(RESULT_PATH, 'proteins.txt'), 'w') as filin:
        for pid in processed_protein_ids:
            filin.write(f'{pid}\n')

    # Generate result pages if requested
    if args.generate_html:
        logging.info("Generating result web pages...")
        # Create a modified FASTA file with generated IDs
        modified_fasta_path = os.path.join(unique_output_dir, 'modified_sequences.fasta')
        with open(modified_fasta_path, 'w') as f:
            for label, seq in zip(labels, seqs):
                f.write(f">{label}\n{seq}\n")

        output_html_dir = os.path.join(unique_output_dir, 'result_pages')
        result_page_generator.generate_result_pages(
            results_dict, modified_fasta_path, output_html_dir, predictions_dir=RESULT_PATH)
        # Generate the results overview page
        generate_results_overview_page(
            labels=labels,
            seqs=seqs,
            id_mapping=id_mapping,
            results_dict=results_dict,
            unique_output_dir=unique_output_dir,
            start_time=start_time,
            run_id=run_id
        )
        logging.info("Result web pages generated.")

    end_time = time.time()
    logging.info(f"Total runtime: {int(end_time - start_time)} seconds")
    logging.info(f"Results saved in: {unique_output_dir}")


if __name__ == '__main__':
    main()