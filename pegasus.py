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
import shutil
import http.server
import socketserver
import threading

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
        "-m", "--models_dir",
        type=str,
        default=os.environ.get('MODELS_DIR', 'models'),
        help="Path to the directory containing pre-trained models."
    )
    parser.add_argument(
        "-o", "--output_dir",
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
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "-t", "--toks_per_batch",
        type=int,
        default=2048,
        help="Tokens per batch to use during embedding generation. Default is 2048."
    )
    parser.add_argument(
        "-g", "--generate_html",
        action='store_true',
        help="Generate result web pages for each protein."
    )
    parser.add_argument(
        "-k", "--keep_embeddings",
        action='store_true',
        help="Keep the LLM raw embeddings in OUTPUT_EMBEDDINGS directory after being used. By default, the directory is deleted after being used."
    )
    parser.add_argument(
        "-a", "--aligned_fasta",
        action="store_true",
        help="Input protein sequences are aligned or not",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length allowed for processing. Sequences longer than this will be skipped. Default is 2048."
    )
    parser.add_argument('--serve', action='store_true', help='Start an HTTP server to serve the result pages.')
    parser.add_argument('--host', default='localhost', help='Hostname to use when serving the result pages. Default is "localhost".')
    parser.add_argument('--port', type=int, default=8000, help='Port to use when serving the result pages. Default is 8000.')
    return parser.parse_args()

def start_http_server(output_dir='result_pages', hostname='localhost', port=8000):
    Handler = http.server.SimpleHTTPRequestHandler

    # Change the current directory to the output directory
    os.chdir(output_dir)

    # Start the server in a separate thread
    def serve():
        with socketserver.TCPServer((hostname, port), Handler) as httpd:
            print(f"Serving HTTP on http://{hostname}:{port}/results_overview.html (press Ctrl+C to stop)")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("Shutting down the server...")
            httpd.server_close()

    server_thread = threading.Thread(target=serve)
    server_thread.daemon = True  # Allows program to exit even if thread is running
    server_thread.start()

    # Keep the main thread alive to keep the server running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down the server...")


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


def validate_fasta(fasta_path, aligned_fasta):
    """
    Validate the FASTA file and extract sequences and labels.

    Args:
        fasta_path (str): Path to the FASTA file.
        aligned_fasta (bool): Flag indicating if sequences are aligned.

    Returns:
        tuple: A tuple containing a list of generated labels, a list of sequences, and a mapping dictionary.
    """
    if not os.path.isfile(fasta_path):
        logging.error(f"The file '{fasta_path}' does not exist.")
        sys.exit(1)

    allowed_letters = set("ACDEFGHIKLMNPQRSTVWY")
    allowed_letters_alignment = set("ACDEFGHIKLMNPQRSTVWY-")
    sequences = []
    labels = []
    id_mapping = {}

    try:
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            seq = str(record.seq).upper()
            if aligned_fasta and not set(seq).issubset(allowed_letters_alignment):
                logging.error(f"Invalid characters found in aligned sequence '{record.id}'. Allowed amino acids are: {allowed_letters_alignment}")
                sys.exit(1)
            elif not aligned_fasta and not set(seq).issubset(allowed_letters):
                logging.error(f"Invalid characters found in sequence '{record.id}'. Allowed amino acids are: {allowed_letters}")
                if set(seq).issubset(allowed_letters_alignment):
                    logging.info("If your sequences are aligned, please add -a or --aligned argument.")
                sys.exit(1)
            sequences.append(seq)
            generated_id = f"P{i+1}"
            labels.append(generated_id)
            id_mapping[generated_id] = record.description
    except Exception as e:
        logging.error(f"Error parsing FASTA file: {e}")
        sys.exit(1)

    if aligned_fasta:
        if len(set(len(seq) for seq in sequences)) != 1:
            logging.error("All sequences must be of the same length for aligned FASTA.")
            sys.exit(1)
        # Check that there is more than one sequence
        if len(sequences) <= 1:
            logging.error("Aligned FASTA requires more than one sequence.")
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
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
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


def embed_ankh(seqs, labels, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using Ankh models.

    Args:
        seqs (list): List of sequences.
        tokenizer: Tokenizer for the model.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        dict: Dictionary mapping labels to embeddings.
    """
    results = {}
    dataset = FastaBatchedDataset.from_list(seqs, labels, is_t5=False)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches)
    with torch.no_grad():
        for batch_labels, batch_strs in tqdm(data_loader, desc="Generating embeddings with Ankh models"):
            sequences = [list(seq) for seq in batch_strs]
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
                label = batch_labels[j]
                results[label] = clean_emb
    return results


def embed_prot_t5(seqs, labels, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using ProtT5 model.

    Args:
        seqs (list): List of sequences.
        tokenizer: Tokenizer for the model.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        dict: Dictionary mapping labels to embeddings.
    """
    results = {}
    dataset = FastaBatchedDataset.from_list(seqs, labels, is_t5=True)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches)
    with torch.no_grad():
        for batch_labels, batch_strs, batch_toks in tqdm(data_loader, desc="Generating embeddings with ProtT5"):
            ids = tokenizer.batch_encode_plus(batch_toks, add_special_tokens=True, padding=True)
            input_ids = torch.tensor(ids['input_ids']).to(device)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            embedding = encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = embedding.last_hidden_state.cpu()
            for i in range(len(embeddings)):
                seq_len = (attention_mask[i] == 1).sum()
                seq_emb = embeddings[i][:seq_len-1]
                label = batch_labels[i]
                results[label] = seq_emb
    return results


def embed_esm2(seqs, labels, tokenizer, encoder, device, toks_per_batch=3072):
    """
    Generate embeddings using ESM2 model.

    Args:
        seqs (list): List of sequences.
        tokenizer: Batch converter obtained from the alphabet.
        encoder: Encoder model.
        device (torch.device): Computation device.
        toks_per_batch (int): Tokens per batch.

    Returns:
        dict: Dictionary mapping labels to embeddings.
    """
    results = {}
    dataset = FastaBatchedDataset.from_list(seqs, labels, is_t5=False)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    batch_converter = tokenizer  # tokenizer is actually the batch_converter
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=batches)
    with torch.no_grad():
        for batch_labels, batch_strs, tokens in tqdm(data_loader, desc="Generating embeddings with ESM2"):
            tokens = tokens.to(device)
            outputs = encoder(tokens, repr_layers=[33], return_contacts=False)
            representations = outputs['representations'][33].cpu()
            for j, seq_emb in enumerate(representations):
                seq_len = len(batch_strs[j])
                seq_emb = seq_emb[1:seq_len+1]
                label = batch_labels[j]
                results[label] = seq_emb
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
    # For each model, generate embeddings
    for model_name in models_list:
        # Load model
        device_choice = model_device_map.get(model_name, default_device)
        device = torch.device("cuda" if device_choice == 'gpu' and torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model and tokenizer for {model_name} on {device} device ...")
        model, tokenizer = load_model_and_tokenizer(model_name, device, models_dir)
        
        logging.info(f"Generating embeddings using {model_name} on device {device}...")
        if model_name in ['ankh_base', 'ankh_large']:
            embeddings_dict = embed_ankh(seqs, labels, tokenizer, model, device, toks_per_batch)
        elif model_name == 'prot_t5_xl_uniref50':
            embeddings_dict = embed_prot_t5(seqs, labels, tokenizer, model, device, toks_per_batch)
        elif model_name == 'esm2_t36_3B_UR50D':
            embeddings_dict = embed_esm2(seqs, labels, tokenizer, model, device, toks_per_batch)
        else:
            raise ValueError(f"Unknown model {model_name}")

        # Save embeddings
        output_dir = os.path.join(output_embeddings_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving embeddings for {model_name}...")
        for label in labels:
            emb = embeddings_dict.get(label)
            if emb is None:
                logging.warning(f"No embedding found for label '{label}' in model '{model_name}'. Skipping.")
                continue
            result = {
                "label": label,
                "embedding": emb
            }
            output_file = os.path.join(output_dir, f"{label}.pt")
            torch.save(result, output_file)

        # Unload model and tokenizer to free up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()


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


def predict_metrics_for_proteins(labels, seqs, aligned_fasta, available_X, available_metrics, model_path, default_device, model_device_map, gapped_seqs=None):
    """
    Load data and make predictions for a list of proteins.

    Args:
        labels (iterable): Names of protein IDs.
        seqs (list): List of sequences.
        aligned_fasta (bool): Flag indicating if sequences are aligned.
        available_X (list): Paths to the embeddings.
        available_metrics (list): Paths to the folders with model data for each metric.
        model_path (list): Names of each model.
        default_device (str): Default computation device.
        model_device_map (dict): Mapping of model names to devices.
        gapped_seqs (list): If aligned_fasta is True, contains the sequences with the gaps.

    Returns:
        tuple: Dictionary of results, Dictionary of aligned results (if aligned_fasta is True), and list of processed protein IDs.
    """
    results_dict = {}
    processed_protein_ids = []
    model_cache = {}
    
    prot_id_to_seq = dict(zip(labels, seqs))

    device_choice = model_device_map.get("pegasus", default_device)
    device = torch.device("cuda" if device_choice == 'gpu' and torch.cuda.is_available() else "cpu")
    logging.info(f"Predict metrics on device {device}...")

    for prot_id in tqdm(labels, desc='Predicting metrics'):
        # Load embedding to get length
        embedding_path = os.path.join(available_X[0], f"{prot_id}.pt")
        if not os.path.exists(embedding_path):
            logging.warning(f"Embedding file '{embedding_path}' does not exist. Skipping protein '{prot_id}'.")
            continue  # Skip if embedding does not exist

        embedding_data = torch.load(embedding_path)
        length = embedding_data["embedding"].shape[0]
        logging.debug(f"Protein ID: {prot_id}, Length: {length}")

        # Verify that sequence is not too long
        if length > SEQ_LENGTH_THRESHOLD:
            logging.warning(f"Protein '{prot_id}' exceeds the sequence length threshold. Skipping.")
            continue  # Skip if protein is too long

        # Load all embeddings for prot_id
        embeddings = {}
        embd_dims = {}
        for embedding_dir, model_name in zip(available_X, model_path):
            embedding_file = os.path.join(embedding_dir, f"{prot_id}.pt")
            if not os.path.exists(embedding_file):
                logging.warning(f"Embedding file '{embedding_file}' does not exist for protein '{prot_id}'. Skipping this embedding.")
                continue  # Skip if embedding file does not exist
            X = torch.load(embedding_file)["embedding"]
            embd_dim = X.shape[1]
            embd_dims[embedding_dir] = embd_dim

            # Pad embeddings to SEQ_LENGTH_THRESHOLD
            X_padded = np.zeros((SEQ_LENGTH_THRESHOLD, embd_dim))
            X_padded[:len(X), :] = X
            X_padded = torch.from_numpy(X_padded.astype(np.float32)).to(device)
            embeddings[embedding_dir] = X_padded

        if not embeddings:
            logging.warning(f"No embeddings loaded for protein '{prot_id}'. Skipping.")
            continue  # Skip if no embeddings loaded

        # Initialize metric dictionaries
        metric_dict = {}
        # Dictionary to collect predictions for each metric
        metric_to_predictions = {'RMSF': [], 'PHI': [], 'PSI': [], 'LDDT': []}
        # Dictionary to store all individual predictions
        all_predictions = {}

        # Loop over each metric
        for metric_path, metric_name in zip(available_metrics, ['RMSF', 'PHI', 'PSI', 'LDDT']):
            # Loop over each embedding/model
            for embedding_dir, model_name in zip(available_X, model_path):
                if embedding_dir not in embeddings:
                    continue  # Skip if embedding not loaded
                X_padded = embeddings[embedding_dir]
                embd_dim = embd_dims[embedding_dir]

                # Load or retrieve model from cache
                key = (metric_path, model_name, embd_dim)
                if key not in model_cache:
                    # Load model based on metric and embedding dimension
                    if metric_path == RMSF_MODEL_PATH or metric_path == LDDT_MODEL_PATH:
                        model = CONV_3L(embd_dim)
                    else:
                        model = CONV_2(embd_dim)
                    path = os.path.join(metric_path, model_name)
                    if not os.path.exists(path):
                        logging.warning(f"Model file '{path}' does not exist. Skipping this model.")
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
                
                # Append prediction to the all_predictions dictionary
                model_short_name = os.path.splitext(model_name)[0]
                all_predictions[f'{metric_name}_{model_short_name}'] = predictions

                # Collect predictions per metric for statistical analysis
                metric_to_predictions[metric_name].append(predictions)

            # Compute mean and standard deviation for the current metric
            if metric_to_predictions[metric_name]:
                metric_values = np.array(metric_to_predictions[metric_name])  # Shape: (n_models, seq_length)
                mean_of_predictions = np.mean(metric_values, axis=0)
                std_of_predictions = np.std(metric_values, axis=0)
            else:
                mean_of_predictions = np.zeros(length, dtype='float64')
                std_of_predictions = np.zeros(length, dtype='float64')

            # Update metric_dict with mean and std
            metric_dict[metric_name] = mean_of_predictions
            metric_dict[f'{metric_name}_std'] = std_of_predictions
             
        # Write the mean of each metric for each protein in a TSV file
        sequence = prot_id_to_seq[prot_id]
        mean_RMSF = metric_dict['RMSF']
        std_RMSF = metric_dict['RMSF_std']
        mean_Phi = metric_dict['PHI']
        std_Phi = metric_dict['PHI_std']
        mean_Psi = metric_dict['PSI']
        std_Psi = metric_dict['PSI_std']
        mean_LDDT = metric_dict['LDDT']
        std_LDDT = metric_dict['LDDT_std']
        df_dl = pd.DataFrame(list(zip(sequence, mean_RMSF, std_RMSF, mean_Phi, std_Phi, mean_Psi, std_Psi, mean_LDDT, std_LDDT)),
                            columns=["res", "mean_RMSF", "std_RMSF", "mean_STD_PHI", "std_STD_PHI", "mean_STD_PSI", "std_STD_PSI", "mean_MEAN_LDDT", "std_MEAN_LDDT"])
        df_dl = df_dl.round(3)
        df_dl.to_csv(os.path.join(RESULT_PATH, f'{prot_id}_predictions.tsv'), sep="\t", index=False)

        # Save mean results into the results dictionary
        processed_protein_ids.append(prot_id)
        results_dict[prot_id] = metric_dict

        # Save per-embedding predictions into a TSV file
        df = pd.DataFrame(all_predictions)
        df = df.round(3)
        df.to_csv(os.path.join(RESULT_PATH, f'{prot_id}_raw.tsv'), sep='\t')


    # Clear model cache and free up GPU memory
    for model in model_cache.values():
        del model
    model_cache.clear()
    torch.cuda.empty_cache()

    # Create the aligned results dictionary if sequences are aligned
    if aligned_fasta:
        # Map protein IDs to sequences with gaps
        prot_id_to_seq = dict(zip(labels, gapped_seqs))
        results_dict_aligned = {}
        for prot_id, metric_dict in results_dict.items():
            sequence = prot_id_to_seq.get(prot_id, "")
            if not sequence:
                logging.warning(f"No sequence found for protein ID '{prot_id}'. Skipping alignment.")
                continue

            aligned_metrics = {}
            for metric_name, metric_values in metric_dict.items():
                # Insert None for positions with gaps ('-')
                aligned_metric_values_full = []
                metric_iter = iter(metric_values)
                for aa in sequence:
                    if aa != '-':
                        try:
                            val = next(metric_iter)
                        except StopIteration:
                            raise ValueError(f"Ran out of values while processing aligned protein '{prot_id}'.")
                        aligned_metric_values_full.append(val)
                    else:
                        aligned_metric_values_full.append(None)
                aligned_metrics[metric_name] = aligned_metric_values_full

            results_dict_aligned[prot_id] = aligned_metrics
    else:
        results_dict_aligned = None

    return results_dict, results_dict_aligned, processed_protein_ids


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


def generate_results_overview_page(labels, seqs, id_mapping, results_dict, results_dict_aligned, unique_output_dir, start_time, run_id, aligned_fasta):
    """
    Generate the results overview page with comparison functionality.
    """
    # Compute job duration
    end_time = time.time()
    job_duration = int(end_time - start_time)
    job_duration_str = format_duration(job_duration)

    # Prepare data for the overview page
    headers = [id_mapping[label] for label in labels]
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
        protein_ids=labels,
        results_dict=results_dict,
        results_dict_aligned=results_dict_aligned,
        output_dir=output_html_dir,
        aligned_fasta=aligned_fasta
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
    
    # Are sequences aligned?
    aligned_fasta = args.aligned_fasta

    # Validate and parse FASTA file
    labels, seqs, id_mapping = validate_fasta(args.input_fasta, aligned_fasta)
    gapped_seqs = None
    if aligned_fasta:
        gapped_seqs = seqs.copy()
        seqs = [seq.replace("-", "") for seq in seqs]

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
    SEQ_LENGTH_THRESHOLD = args.max_seq_length
    
    toks_per_batch = args.toks_per_batch
    
    if SEQ_LENGTH_THRESHOLD > toks_per_batch:
        logging.error(f"Max sequence length (--max_seq_length) {SEQ_LENGTH_THRESHOLD} cannot be bigger than the max tokens per batch (--toks_per_batch) {toks_per_batch}")
        sys.exit(1)

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

    # Fit the model and make predictions
    logging.info("Starting predictions...")
    results_dict, results_dict_aligned, processed_protein_ids = predict_metrics_for_proteins(
        labels, seqs, aligned_fasta, X_paths, metrics, models_names, default_device, model_device_map, gapped_seqs=gapped_seqs)
    logging.info("Predictions completed.")

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
            results_dict, modified_fasta_path, id_mapping, output_html_dir, predictions_dir=RESULT_PATH)
        
        # Pass the aligned results_dict_aligned to generate_results_overview_page
        generate_results_overview_page(
            labels=labels,
            seqs=seqs if not aligned_fasta else gapped_seqs,  # Original sequences with gaps if aligned_fasta is True
            id_mapping=id_mapping,
            results_dict=results_dict,
            results_dict_aligned=results_dict_aligned,
            unique_output_dir=unique_output_dir,
            start_time=start_time,
            run_id=run_id,
            aligned_fasta=aligned_fasta
        )
        logging.info("Result web pages generated.")

    # Delete embeddings directory if not keeping embeddings
    if not args.keep_embeddings:
        logging.info("Deleting embeddings directory as per default setting.")
        shutil.rmtree(OUTPUT_EMBEDDINGS)

    end_time = time.time()
    logging.info(f"Total runtime: {int(end_time - start_time)} seconds")
    logging.info(f"Results saved in: {unique_output_dir}")

    if args.generate_html and args.serve:
        start_http_server(output_dir=output_html_dir, hostname=args.host, port=args.port)

if __name__ == '__main__':
    main()