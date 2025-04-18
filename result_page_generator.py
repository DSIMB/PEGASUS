#!/usr/bin/env python3
"""
Result Page Generator Module

This module generates individual HTML pages with results for each input protein.
It provides functions to create plots and other outputs based on the prediction results.
"""

import os
import pickle

import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import plotly.graph_objects as go
from Bio import SeqIO

# Mapping dictionaries for amino acid codes
AA_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
    'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
    'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
}

AA_1TOFULL = {
    'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartic Acid', 'E': 'Glutamic Acid',
    'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
    'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine', 'N': 'Asparagine',
    'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine', 'S': 'Serine',
    'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan', 'Y': 'Tyrosine'
}


def load_tsv(protein_id, predictions_dir='predictions'):
    """
    Load TSV file with per-embedding predictions for a given protein.

    Args:
        protein_id (str): The protein identifier.
        predictions_dir (str): The directory where prediction files are stored.

    Returns:
        list: A list of numpy arrays, each containing per-embedding predictions for a metric.
    """
    filepath = os.path.join(predictions_dir, f'{protein_id}_raw.tsv')
    df = pd.read_csv(filepath, sep='\t')

    # Check if DataFrame has at least 2 columns
    if df.shape[1] < 2:
        raise ValueError(f"TSV file {filepath} must have at least two columns.")

    # Extract metrics (excluding first column, assuming it's an index or position)
    all_metrics = df.iloc[:, 1:].values

    # Determine number of metrics
    num_metrics = 4  # Assuming 4 metrics
    num_embeddings = all_metrics.shape[1]

    if num_embeddings % num_metrics != 0:
        raise ValueError("Number of embeddings is not divisible by the number of metrics.")

    # Split metrics into list
    results_by_metric = np.array_split(all_metrics, num_metrics, axis=1)

    return results_by_metric


def draw_interactive_plot(result_dict, protein_id, output_dir, results_by_metric):
    """
    Draw interactive Plotly plots for each predicted metric and save as HTML files.

    Args:
        result_dict (dict): Dictionary containing results with proteins as keys.
        protein_id (str): Protein identifier.
        output_dir (str): Directory to save HTML output files.
        results_by_metric (list): List of numpy arrays with individual embedding predictions.

    Returns:
        None
    """
    COLORS = ['firebrick', 'steelblue', 'forestgreen', 'gold']
    METRICS = ['RMSF', 'PHI', 'PSI', 'LDDT']
    UNITS = ['Å', '°', '°', '']


    #RMSF
    metric = "RMSF"

    matrix = results_by_metric[0]

    # Get the mean result for the metric
    result = result_dict[protein_id][metric]

    #Convert RMSF from nm to Angstrom
    result = result * 10
        
    result = np.around(result, 3)
    x = np.arange(1, len(result)+1)


    # Calculate standard deviation across embeddings
    metric_std = np.std(matrix, axis=1)
    
    #Convert RMSF from nm to Angstrom
    metric_std = metric_std * 10
    
    metric_std = np.around(metric_std, 3)


    # Create figure
    fig = go.Figure()

    # Create main metric trace
    fig.add_trace(go.Scatter(
        x=x, y=result,
        line=dict(color='rgba(221, 97, 74, 1)'),
        mode="lines", name="RMSF"
    ))

    fig.update_traces(customdata=metric_std)
    fig["data"][0]["hovertemplate"] = "%{x}<br>RMSF = %{y} ± %{customdata} Å<extra></extra>"
    
    fig.update_layout(hoverlabel=dict(bgcolor="white"),
                      hoverlabel_font_color="black")

    # Add confidence interval (mean ± std)
    fig.add_traces([
        go.Scatter(
            x=x, y=result+metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            showlegend=False, hoverinfo='none'
        ),
        go.Scatter(
            x=x, y=result-metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            name='SD',
            fill='tonexty', fillcolor='rgba(221, 97, 74, 0.2)',
            hoverinfo='none'
        )
    ])

    # Update layout parameters
    fig.update_layout(
        xaxis_title="<b>Position</b>",
        yaxis_title="<b>Pred. RMSF (Å)</b>",
        yaxis=dict(range=[0, np.nanmax(result+metric_std) * 1.05]),
        showlegend=False,
        margin=dict(l=10, r=10, b=30, t=30, pad=4),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        template="plotly_white",
        hovermode="x unified"
    )

    # Save plot to HTML file
    output_filepath = os.path.join(output_dir, f"{protein_id}_{metric}_pred.html")
    fig.write_html(
        output_filepath,
        full_html=False, auto_play=False,
        include_plotlyjs=False, config={"responsive": True},
        default_width="100%", default_height="100%"
    )




    #Phi
    metric = "PHI"

    matrix = results_by_metric[1]

    # Get the mean result for the metric
    result = result_dict[protein_id][metric] 
    result = np.around(result, 3)
    x = np.arange(1, len(result)+1)

    # Calculate standard deviation across embeddings
    metric_std = np.std(matrix, axis=1)
    metric_std = np.around(metric_std, 3)


    # Create figure
    fig = go.Figure()

    # Create main metric trace
    fig.add_trace(go.Scatter(
        x=x, y=result,
        line=dict(color='rgba(115, 165, 128, 1)'),
        mode="lines", name=f"Std. Phi"
    ))

    fig.update_traces(customdata=metric_std)
    fig["data"][0]["hovertemplate"] = "%{x}<br>Std. Phi = %{y} ± %{customdata} °<extra></extra>"
    
    fig.update_layout(hoverlabel=dict(bgcolor="white"),
                      hoverlabel_font_color="black")

    # Add confidence interval (mean ± std)
    fig.add_traces([
        go.Scatter(
            x=x, y=result+metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            showlegend=False, hoverinfo='none'
        ),
        go.Scatter(
            x=x, y=result-metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            name='SD',
            fill='tonexty', fillcolor='rgba(115, 165, 128, 0.2)',
            hoverinfo='none'
        )
    ])

    # Update layout parameters
    fig.update_layout(
        xaxis_title="<b>Position</b>",
        yaxis_title="<b>Pred. Std. Phi (°)</b>",
        yaxis=dict(range=[0, np.nanmax(result+metric_std) * 1.05]),
        showlegend=False,
        margin=dict(l=10, r=10, b=30, t=30, pad=4),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        template="plotly_white",
        hovermode="x unified"
    )

    # Save plot to HTML file
    output_filepath = os.path.join(output_dir, f"{protein_id}_{metric}_pred.html")
    fig.write_html(
        output_filepath,
        full_html=False, auto_play=False,
        include_plotlyjs=False, config={"responsive": True},
        default_width="100%", default_height="100%"
    )




    #Psi
    metric = "PSI"

    matrix = results_by_metric[2]

    # Get the mean result for the metric
    result = result_dict[protein_id][metric] 
    result = np.around(result, 3)
    x = np.arange(1, len(result)+1)

    # Calculate standard deviation across embeddings
    metric_std = np.std(matrix, axis=1)
    metric_std = np.around(metric_std, 3)


    # Create figure
    fig = go.Figure()

    # Create main metric trace
    fig.add_trace(go.Scatter(
        x=x, y=result,
        line=dict(color='rgba(127, 106, 147, 1)'),
        mode="lines", name=f"Std. Psi"
    ))

    fig.update_traces(customdata=metric_std)
    fig["data"][0]["hovertemplate"] = "%{x}<br>Std. Psi = %{y} ± %{customdata} °<extra></extra>"
    
    fig.update_layout(hoverlabel=dict(bgcolor="white"),
                      hoverlabel_font_color="black")

    # Add confidence interval (mean ± std)
    fig.add_traces([
        go.Scatter(
            x=x, y=result+metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            showlegend=False, hoverinfo='none'
        ),
        go.Scatter(
            x=x, y=result-metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            name='SD',
            fill='tonexty', fillcolor='rgba(127, 106, 147, 0.2)',
            hoverinfo='none'
        )
    ])

    # Update layout parameters
    fig.update_layout(
        xaxis_title="<b>Position</b>",
        yaxis_title="<b>Pred. Std. Psi (°)</b>",
        yaxis=dict(range=[0, np.nanmax(result+metric_std) * 1.05]),
        showlegend=False,
        margin=dict(l=10, r=10, b=30, t=30, pad=4),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        template="plotly_white",
        hovermode="x unified"
    )

    # Save plot to HTML file
    output_filepath = os.path.join(output_dir, f"{protein_id}_{metric}_pred.html")
    fig.write_html(
        output_filepath,
        full_html=False, auto_play=False,
        include_plotlyjs=False, config={"responsive": True},
        default_width="100%", default_height="100%"
    )



    #Mean LDDT
    metric = "LDDT"

    matrix = results_by_metric[3]

    # Get the mean result for the metric
    result = result_dict[protein_id][metric] 
    result = np.around(result, 3)
    x = np.arange(1, len(result)+1)

    # Calculate standard deviation across embeddings
    metric_std = np.std(matrix, axis=1)
    metric_std = np.around(metric_std, 3)


    # Create figure
    fig = go.Figure()

    # Create main metric trace
    fig.add_trace(go.Scatter(
        x=x, y=result,
        line=dict(color='rgba(213, 160, 33, 1)'),
        mode="lines", name=f"Mean LDDT"
    ))

    fig.update_traces(customdata=metric_std)
    fig["data"][0]["hovertemplate"] = "%{x}<br>Mean LDDT = %{y} ± %{customdata}<extra></extra>"
    
    fig.update_layout(hoverlabel=dict(bgcolor="white"),
                      hoverlabel_font_color="black")

    # Add confidence interval (mean ± std)
    fig.add_traces([
        go.Scatter(
            x=x, y=result+metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            showlegend=False, hoverinfo='none'
        ),
        go.Scatter(
            x=x, y=result-metric_std,
            mode='lines', line_color='rgba(0,0,0,0)',
            name='SD',
            fill='tonexty', fillcolor='rgba(213, 160, 33, 0.2)',
            hoverinfo='none'
        )
    ])

    # Update layout parameters
    fig.update_layout(
        xaxis_title="<b>Position</b>",
        yaxis_title="<b>Pred. Mean LDDT</b>",
        yaxis=dict(range=[0, np.nanmax(result+metric_std) * 1.05]),
        showlegend=False,
        margin=dict(l=10, r=10, b=30, t=30, pad=4),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        template="plotly_white",
        hovermode="x unified"
    )

    # Save plot to HTML file
    output_filepath = os.path.join(output_dir, f"{protein_id}_{metric}_pred.html")
    fig.write_html(
        output_filepath,
        full_html=False, auto_play=False,
        include_plotlyjs=False, config={"responsive": True},
        default_width="100%", default_height="100%"
    )



def draw_interactive_plot_all(result_dict, protein_id, output_dir):
    """
    Draw interactive Plotly plot with all normalized predicted metrics and save as an HTML file.

    Args:
        result_dict (dict): Dictionary containing results with proteins as keys.
        protein_id (str): Protein identifier.
        output_dir (str): Directory to save HTML output file.

    Returns:
        None
    """
    METRICS = ['RMSF', 'PHI', 'PSI', 'LDDT']

    # Retrieve and normalize metrics
    normalized_metrics = {}
    for metric in METRICS:
        values = result_dict[protein_id][metric]
        normalized_values = np.interp(values, (values.min(), values.max()), (0, 1))
        normalized_metrics[metric] = np.around(normalized_values, 3)

    # Prepare x-axis values
    x = np.arange(1, len(next(iter(normalized_metrics.values()))) + 1)

    # Create figure
    fig = go.Figure()

    # Add traces for each metric
    fig.add_trace(go.Scatter(
            x=x, y=normalized_metrics['RMSF'],
            line=dict(color='rgba(221, 97, 74, 1)'),
            mode="lines", name=f"RMSF (Å)"
        ))

    fig.add_trace(go.Scatter(
                x=x, y=normalized_metrics['PHI'],
                line=dict(color='rgba(115, 165, 128, 1)'),
                mode="lines", name=f"Std. Phi (°)"
            ))

    fig.add_trace(go.Scatter(
                x=x, y=normalized_metrics['PSI'],
                line=dict(color='rgba(127, 106, 147, 1)'),
                mode="lines", name=f"Std. Psi (°)"
            ))

    fig.add_trace(go.Scatter(
                x=x, y=normalized_metrics['LDDT'],
                line=dict(color='rgba(213, 160, 33, 1)'),
                mode="lines", name=f"Mean LDDT"
            ))


    # Update layout parameters
    fig.update_layout(
        xaxis_title="<b>Position</b>",
        yaxis_title="<b>Normalised metrics</b>",
        yaxis=dict(range=[0, 1.05]),
        showlegend=True,
        margin=dict(l=10, r=10, b=30, t=30, pad=4),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        template="plotly_white",
        hovermode="x unified"
    )

    # Save plot to HTML file
    output_filepath = os.path.join(output_dir, f"{protein_id}_ALL_pred.html")
    fig.write_html(
        output_filepath,
        full_html=False, auto_play=False,
        include_plotlyjs=False, config={"responsive": True},
        default_width="100%", default_height="100%"
    )



def write_result_page(result_dict, protein_realname, protein_id, output_dir, sequence):
    """
    Write the full results page for the PEGASUS model as an HTML file.

    Args:
        result_dict (dict): Dictionary containing results with proteins as keys.
        protein_realname (str): The real name or header of the protein.
        protein_id (str): Protein identifier.
        output_dir (str): Directory to save the HTML output file.
        sequence (str): Amino acid sequence of the protein.

    Returns:
        None
    """
    # Normalize and round metric values for display
    metrics = ['RMSF', 'PHI', 'PSI', 'LDDT']
    normalized_metrics = {}
    rounded_metrics = {}
    for metric in metrics:
        values = result_dict[protein_id][metric]
        normalized_values = np.interp(values, (values.min(), values.max()), (0, 1))
        normalized_metrics[metric] = np.around(normalized_values, 3)
        rounded_metrics[metric] = np.around(values, 3)

    # Generate the sequenceTrack content
    sequence_track_displays = []
    for idx, res in enumerate(sequence, start=1):
        display = f"""{{displayType:"sequence", displayId:"cs{idx}", displayData:[{{begin:{idx}, value:"{res}", featureId:"[{AA_1TO3.get(res, '')} : {AA_1TOFULL.get(res, '')}]"}}]}},\n"""
        sequence_track_displays.append(display)
    sequence_track_config = ''.join(sequence_track_displays)



    # Generate the area tracks for each metric
    area_tracks = ''
    colors = ['#DD614A', '#73A580', '#7F6A93', '#D5A021']
    

    #RMSF
    metric = metrics[0]
    color = colors[0]
    
    track_data_entries = ''
    for idx, (norm_value, value) in enumerate(zip(normalized_metrics[metric], rounded_metrics[metric]*10), start=1):
        track_data_entries += f"{{begin:{idx}, value:{norm_value}, featureId:\"value: {value:.2f} Å\"}},\n"
    
    area_track = f"""
    const {metric.lower()}Track = {{
        trackId: "{metric.lower()}Track",
        trackHeight: 100,
        trackColor: "#F9F9F9",
        displayType: "area",
        nonEmptyDisplay: true,
        interpolationType: "cardinal",
        displayColor: "{color}",
        rowTitle: "RMSF (Å)",
        fitTitleWidth: true,
        trackData: [
            {track_data_entries}
        ]
    }};
    """
    area_tracks += area_track


    #Phi
    metric = metrics[1]
    color = colors[1]
    
    track_data_entries = ''
    for idx, (norm_value, value) in enumerate(zip(normalized_metrics[metric], rounded_metrics[metric]), start=1):
        track_data_entries += f"{{begin:{idx}, value:{norm_value}, featureId:\"value: {value:.2f} °\"}},\n"
    
    area_track = f"""
    const {metric.lower()}Track = {{
        trackId: "{metric.lower()}Track",
        trackHeight: 100,
        trackColor: "#F9F9F9",
        displayType: "area",
        nonEmptyDisplay: true,
        interpolationType: "cardinal",
        displayColor: "{color}",
        rowTitle: "Std. Phi (°)",
        fitTitleWidth: true,
        trackData: [
            {track_data_entries}
        ]
    }};
    """
    area_tracks += area_track

    

    #Psi
    metric = metrics[2]
    color = colors[2]
    
    track_data_entries = ''
    for idx, (norm_value, value) in enumerate(zip(normalized_metrics[metric], rounded_metrics[metric]), start=1):
        track_data_entries += f"{{begin:{idx}, value:{norm_value}, featureId:\"value: {value:.2f} °\"}},\n"
    
    area_track = f"""
    const {metric.lower()}Track = {{
        trackId: "{metric.lower()}Track",
        trackHeight: 100,
        trackColor: "#F9F9F9",
        displayType: "area",
        nonEmptyDisplay: true,
        interpolationType: "cardinal",
        displayColor: "{color}",
        rowTitle: "Std. Psi (°)",
        fitTitleWidth: true,
        trackData: [
            {track_data_entries}
        ]
    }};
    """
    area_tracks += area_track



    #LDDT
    metric = metrics[3]
    color = colors[3]
    
    track_data_entries = ''
    for idx, (norm_value, value) in enumerate(zip(normalized_metrics[metric], rounded_metrics[metric]), start=1):
        track_data_entries += f"{{begin:{idx}, value:{norm_value}, featureId:\"value: {value:.2f}\"}},\n"
    
    area_track = f"""
    const {metric.lower()}Track = {{
        trackId: "{metric.lower()}Track",
        trackHeight: 100,
        trackColor: "#F9F9F9",
        displayType: "area",
        nonEmptyDisplay: true,
        interpolationType: "cardinal",
        displayColor: "{color}",
        rowTitle: "Mean LDDT",
        fitTitleWidth: true,
        trackData: [
            {track_data_entries}
        ]
    }};
    """
    area_tracks += area_track




    # Define the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Meta tags and page title -->
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>PEGASUS | {protein_realname}</title>
    <link id="favicon" rel="icon" href="https://dsimb.inserm.fr/PEGASUS/images/favicon.png" type="image/png" sizes="16x16">

    <!-- CSS and JS includes -->
    <!-- Bootstrap -->
    <link href="https://dsimb.inserm.fr/PEGASUS/css/bootstrap.min.css" rel="stylesheet">
    <!-- Load Chart.js -->
    <script src="https://dsimb.inserm.fr/PEGASUS/js/d3.v4.js"></script>
    <script src="https://dsimb.inserm.fr/PEGASUS/js/Chart.bundle.min.js"></script>
    <script src="https://dsimb.inserm.fr/PEGASUS/js/chartjs-plugin-labels.js"></script>
    <!-- jQuery -->
    <script src="https://dsimb.inserm.fr/PEGASUS/js/jquery.min.js"></script>
    <link rel="stylesheet" href="https://dsimb.inserm.fr/PEGASUS/css/jquery-ui.min.css"/>
    <script src="https://dsimb.inserm.fr/PEGASUS/js/jquery-ui.min.js"></script>
    <!-- Features viewer -->
    <script src="https://dsimb.inserm.fr/PEGASUS/js/custom_rcsb_saguaro_mini.js"></script>
    <!-- Plotly -->
    <script src="https://dsimb.inserm.fr/PEGASUS/js/plotly-2.12.1.min.js"></script>
    <!-- Custom Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;1,100;1,200;1,300;1,400;1,500;1,600;1,700&display=swap" rel="stylesheet">
    <!-- Custom styles -->
    <link href="https://dsimb.inserm.fr/PEGASUS/css/custom_features.css" rel="stylesheet" />
</head>
<body class="d-flex flex-column min-vh-100">

    <!-- Header -->
    <nav class="navbar navbar-expand-lg navbar-light bg-white fixed-top">
        <div class="container pb-2" id="nav-container">
            <a class="navbar-brand" href="/PEGASUS/index.html">
                <img src="https://dsimb.inserm.fr/PEGASUS/images/PEGASUS_logo.png" alt="Website logo" id="logo" style="transition: 0.4s;height:70px;">
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarResponsive"
                aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"
                style="margin-top: 3px;margin-bottom: 2px">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive" style="margin-top: auto">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="https://dsimb.inserm.fr/PEGASUS/index.html">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="https://dsimb.inserm.fr/PEGASUS/about.html">About</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="https://dsimb.inserm.fr/PEGASUS/example.html">Example</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="https://dsimb.inserm.fr/PEGASUS/contact.html">Contact</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Page Content -->
    <div class="container mt-7">
        <div class="alert alert-success mt-5 mb-0" role="alert">
            <h1 class="alert-heading"> Results </h1>
            <p class="fw-bold mb-0">Query: <span class="fw-normal text-break font-monospace">{protein_realname}</span></p>
            <p class="fw-bold mb-0">Sequence: <span class="fw-normal text-break font-monospace">{sequence}</span></p>
            <p class="fw-bold mb-0">Length: <span class="fw-normal text-break font-monospace">{seq_length}</span></p>
        </div>

        <!-- Summary -->
        <div class="card mt-2 pt-4">
            <h2>  Flexibility Profiles </h2>
            <hr class="mt-2">
            <div class="card-body px-0 mt-3 w-100 mb-2">
                <div id="summary" style="height: 445px"> 
                    <!-- Sequence viewer -->
                    <div id="pfv" class="mb-2"></div>
                </div>
            </div>
        </div>

        <!-- Metric by Metric Predictions -->
        <div class="card mt-2 pt-4">
            <h2> Metric by Metric Predictions </h2>
            <hr class="mt-2">
            <div class="card-body px-0 mt-3 w-100 mb-2">
                <div class="container" id="results-overview">
                    <div class="row">
                        <div class="col-12">
                            <div class="card mt-0" id="flex_rep">
                                <h5 class="card-header"> Flexibility profile </h5>
                                <div class="d-flex justify-content-center">
                                    <div class="ratio" style="--bs-aspect-ratio: 30%;position:relative;" id="plot_rep"></div>
                                </div>
                                <div class="btn-toolbar d-flex justify-content-center pt-2 pb-4" role="toolbar" aria-label="toolbar_rep">
                                    <div>
                                        <input type="radio" class="btn-check" name="btnradio_rep" id="btnradio_rep2" autocomplete="off" checked>
                                        <label class="btn btn-outline-primary me-2" for="btnradio_rep2" onclick="display_rep_rmsf();">RMSF</label>
                                        <input type="radio" class="btn-check" name="btnradio_rep" id="btnradio_rep1" autocomplete="off">
                                        <label class="btn btn-outline-primary me-2" for="btnradio_rep1" onclick="display_rep_phi();">Std. Phi</label>
                                        <input type="radio" class="btn-check" name="btnradio_rep" id="btnradio_rep3" autocomplete="off">
                                        <label class="btn btn-outline-primary me-2" for="btnradio_rep3" onclick="display_rep_psi();">Std. Psi</label>
                                        <input type="radio" class="btn-check" name="btnradio_rep" id="btnradio_rep4" autocomplete="off">
                                        <label class="btn btn-outline-primary me-2" for="btnradio_rep4" onclick="display_rep_lddt();">Mean LDDT</label>
                                        <input type="radio" class="btn-check" name="btnradio_rep" id="btnradio_rep5" autocomplete="off">
                                        <label class="btn btn-outline-primary" for="btnradio_rep5" onclick="display_all();">Comparison</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Metric by Metric Scripts -->
        <script>
            const spinner = `<div class="d-flex justify-content-center h-100 w-100">
                <div class="spinner-border text-primary my-auto" role="status" style="color: #146698 !important;">
                <span class="visually-hidden">Loading...</span>
                </div>
                </div>`;
            function display_rep_phi() {{
                $('#plot_rep').children().css("position", "absolute").css("z-index", "-1").addClass("blur");
                $('#plot_rep').append(spinner);
                $("#plot_rep").load("{protein_id}_PHI_pred.html");
            }};
            function display_rep_rmsf() {{
                $('#plot_rep').children().css("position", "absolute").css("z-index", "-1").addClass("blur");
                $('#plot_rep').append(spinner);
                $("#plot_rep").load("{protein_id}_RMSF_pred.html");
            }};
            function display_rep_psi() {{
                $('#plot_rep').children().css("position", "absolute").css("z-index", "-1").addClass("blur");
                $('#plot_rep').append(spinner);
                $("#plot_rep").load("{protein_id}_PSI_pred.html");
            }};
            function display_rep_lddt() {{
                $('#plot_rep').children().css("position", "absolute").css("z-index", "-1").addClass("blur");
                $('#plot_rep').append(spinner);
                $("#plot_rep").load("{protein_id}_LDDT_pred.html");
            }};
            function display_all() {{
                $('#plot_rep').children().css("position", "absolute").css("z-index", "-1").addClass("blur");
                $('#plot_rep').append(spinner);
                $("#plot_rep").load("{protein_id}_ALL_pred.html");
            }};
        </script>

        <!-- Sequence Viewer Script -->
        <script>
            const sequence = "{sequence}";

            // Adjust track width based on window size
            let trackWidth_start = 1117;
            if (window.matchMedia("(max-width: 1400px)").matches) {{
                trackWidth_start = 1007;
            }}
            if (window.matchMedia("(max-width: 1200px)").matches) {{
                trackWidth_start = 827;
            }}
            if (window.matchMedia("(max-width: 992px)").matches) {{
                trackWidth_start = 587;
            }}
            if (window.matchMedia("(max-width: 768px)").matches) {{
                trackWidth_start = 407;
            }}

            const boardConfigData = {{
                length: sequence.length,
                trackWidth: trackWidth_start,
                rowTitleWidth: 129,
                includeAxis: true,
                includeTooltip: true,
                disableMenu: false,
                range: {{min: 2, max: sequence.length - 1}}
            }};

            // Sequence track configuration
            const sequenceTrack = {{
                trackId: "sequenceTrack",
                trackHeight: 25,
                trackColor: "#F9F9F9",
                displayType: "composite",
                nonEmptyDisplay: true,
                rowTitle: "Sequence",
                displayConfig: [
                    {sequence_track_config}
                ]
            }};

            {area_tracks}

            // Initialize the feature viewer
            const pfv = new RcsbFv.Create({{
                boardConfigData: boardConfigData,
                rowConfigData: [sequenceTrack, rmsfTrack, phiTrack, psiTrack, lddtTrack],
                elementId: "pfv"
            }});

            // Resize feature viewer on window resize
            let trackWidth_current = trackWidth_start;
            function resizeFeatureViewer() {{
                let newTrackWidth;
                if (window.matchMedia("(min-width: 1400px)").matches) {{
                    newTrackWidth = 1117;
                }} else if (window.matchMedia("(min-width: 1200px)").matches) {{
                    newTrackWidth = 1007;
                }} else if (window.matchMedia("(min-width: 992px)").matches) {{
                    newTrackWidth = 827;
                }} else if (window.matchMedia("(min-width: 768px)").matches) {{
                    newTrackWidth = 587;
                }} else {{
                    newTrackWidth = 407;
                }}
                if (newTrackWidth !== trackWidth_current) {{
                    trackWidth_current = newTrackWidth;
                    pfv.updateBoardConfig({{boardConfigData: {{trackWidth: trackWidth_current}}}});
                }}
            }};
            window.addEventListener('resize', resizeFeatureViewer, false);
        </script>

        <!-- Footer -->
        <div id="footer" class="mt-auto"></div>
    </div>

    <!-- Load external scripts -->
    <script>
        $(function() {{
            $("#footer").load("https://dsimb.inserm.fr/PEGASUS/footer.html");
            $("#plot_rep").load("{protein_id}_RMSF_pred.html");
        }});
    </script>

</body>
</html>
'''

    # Format the HTML content with the variables
    html_content = html_template.format(
        protein_realname=protein_realname,
        protein_id=protein_id,
        sequence=sequence,
        seq_length=len(sequence),
        sequence_track_config=sequence_track_config,
        area_tracks=area_tracks
    )

    # Write the HTML content to the file
    output_filepath = os.path.join(output_dir, f"{protein_id}.html")
    with open(output_filepath, 'w') as f_out:
        f_out.write(html_content)



def generate_result_pages(result_dict, fasta_file, id_mapping, output_dir='result_pages', predictions_dir='predictions'):
    """
    Generate result pages for all proteins in the given FASTA file.

    Args:
        result_dict (dict): Dictionary containing results with proteins as keys.
        fasta_file (str): Path to the input FASTA file.
        id_mapping (dict): Dictionary mapping the original protein ids with the unique P{i} generated by Pegasus
        output_dir (str): Directory to save HTML output files.
        predictions_dir (str): Directory where prediction TSV files are stored.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read sequences from FASTA file
    headers = []
    sequences = []
    protein_ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq).upper()
        protein_id = record.id.strip()
        header = id_mapping[protein_id]
        sequences.append(sequence)
        headers.append(header)
        protein_ids.append(protein_id)

    # Process each protein
    for protein_realname, sequence, protein_id in zip(headers, sequences, protein_ids):
        # Load TSV data
        all_results = load_tsv(protein_id, predictions_dir)

        # Generate plots and result page
        draw_interactive_plot(result_dict, protein_id, output_dir, all_results)
        draw_interactive_plot_all(result_dict, protein_id, output_dir)
        write_result_page(result_dict, protein_realname, protein_id, output_dir, sequence)


def write_results_overview_page(job_id, job_duration, date, headers, sequences, protein_ids, results_dict, results_dict_aligned, output_dir, aligned_fasta):
    """
    Generate the results overview HTML page with comparison functionality.
    
    Args:
        job_id (str): Job identifier.
        job_duration (str): Duration of the job.
        date (str): Date of the job.
        headers (list): List of protein headers.
        sequences (list): List of protein sequences.
        protein_ids (list): List of protein IDs.
        results_dict (dict): Dictionary containing results.
        results_dict_aligned (dict or None): Dictionary containing aligned results with None for gaps.
        output_dir (str): Directory to save the overview page.
        aligned_fasta (bool): Flag indicating if sequences are aligned.
    """
    import json

    # Prepare the output file path
    output_filepath = os.path.join(output_dir, "results_overview.html")

    # Generate table rows for each protein
    table_rows = ''
    for i, prot_id in enumerate(protein_ids):
        header = headers[i]
        sequence = sequences[i]
        seq_length = len(sequence)
        table_rows += f'''
            <tr>
                <td style="width: 2%; cursor:pointer;"></td><!-- Checkbox column; leave empty -->
                <td class="details-control" style="width: 2%; cursor:pointer;">
                    <span class="me-2">
                        <!-- Arrow icon -->
                        <svg class="collapse-icon bi bi-caret-right-fill" width="1em" height="1em" fill="currentColor" xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 16 16">
                            <path d="M12 8L6 12V4l6 4z"/>
                        </svg>
                    </span>
                </td>
                <td class="td-ellipsis"><strong>{prot_id}</strong> - {header}</td>
                <td class="text-center" style="width: 8%;">{seq_length}</td>
                <td>{sequence}</td> <!-- Hidden column -->
                <td class="text-center" style="width: 3%;">
                    <a class="btn" role="button" href="../predictions/{prot_id}_predictions.tsv" download="{prot_id}_predictions.tsv">
                        <!-- Download icon -->
                        <svg width="1.2em" xmlns="http://www.w3.org/2000/svg" fill="none"
                             viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="align-text-top">
                             <path stroke-linecap="round" stroke-linejoin="round"
                                   d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15M9 12l3 3m0 0 3-3m-3 3V2.25" />
                        </svg>
                    </a>
                </td>
                <td class="text-center" style="width: 10%;">
                    <a class="btn" href="{prot_id}.html" role="button" target="_blank">
                        <!-- View icon -->
                        <svg width="1.2em" xmlns="http://www.w3.org/2000/svg" fill="none"
                             viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="align-text-top">
                             <path stroke-linecap="round" stroke-linejoin="round"
                                   d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" />
                        </svg>
                    </a>
                </td>
            </tr>
        '''

    # Construct the full HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <!-- Website title -->
    <title>PEGASUS Results Overview</title>
    <link id="favicon" rel="icon" href="https://dsimb.inserm.fr/PEGASUS/images/favicon.png" type="image/png" sizes="16x16">

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

    <link href="https://cdn.datatables.net/v/bs5/dt-2.1.8/b-3.1.2/b-colvis-3.1.2/b-html5-3.1.2/r-3.0.3/sl-2.1.0/datatables.min.css" rel="stylesheet">
    <script src="https://cdn.datatables.net/v/bs5/dt-2.1.8/b-3.1.2/b-colvis-3.1.2/b-html5-3.1.2/r-3.0.3/sl-2.1.0/datatables.min.js"></script>

    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>

    <!-- Custom styles for this template -->
    <link href="https://dsimb.inserm.fr/PEGASUS/css/custom_features.css" rel="stylesheet" />

</head>

<body class="d-flex flex-column min-vh-100">

    <!-- Header -->
    <nav class="navbar navbar-expand-lg navbar-light bg-white fixed-top">
        <div class="container pb-2" id="nav-container">
            <a class="navbar-brand" href="#">
                <img src="https://dsimb.inserm.fr/PEGASUS/images/PEGASUS_logo.png" alt="Website logo" id="logo" style="transition: 0.4s;height:70px;">
            </a>
        </div>
    </nav>

    <!-- Page Content -->
    <div class="container mt-7">
        <div class="alert alert-info" role="alert">
            <h1 class="alert-heading">Results Overview</h1>
            <strong>Job ID:</strong> {job_id}<br>
            <strong>Launch date:</strong> {date}<br>
            <strong>Number of sequences:</strong> {len(protein_ids)}<br>
            <strong>Run time:</strong> {job_duration}<br>
        </div>

        <!-- Summary -->
        <div class="container mt-4 pt-4">
            <h2 class="search-title mb-4">Queries</h2>

            <!-- Compare Button and Informational Text -->
            <span class="d-inline-block mb-3" tabindex="0" data-bs-toggle="tooltip" data-bs-placement="right" data-bs-title="Select at least two proteins of the same length to compare.">
                <button id="compareButton" class="btn btn-success" disabled>
                    Compare Selected Proteins
                </button>
            </span>

            <div id="queries_table_container">
                <table id="queries_table" class="table align-middle table-hover">
                    <thead>
                        <tr>
                            <th class="dt-checkboxes-select-all"></th>
                            <th></th> <!-- Expand/Collapse control column -->
                            <th>Name</th>
                            <th>Length</th>
                            <th>Sequence</th> <!-- Hidden column -->
                            <th>Download</th>
                            <th>Result page</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Comparison Section -->
        <div id="comparison_section">
            <!-- Comparison Plot will be inserted here -->
        </div>

    </div>

    <!-- Footer -->
    <div id="footer" class="mt-auto"></div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Custom JS -->
    <script>
        // Initialize all tooltips on the page
        document.addEventListener('DOMContentLoaded', function () {{
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                return new bootstrap.Tooltip(tooltipTriggerEl);
            }});
        }});
        
        // Embed the aligned_fasta flag as a JavaScript variable
        var aligned_fasta = {str(aligned_fasta).lower()};
    
        // JavaScript code to handle table interactions and comparison functionality

        // Load the appropriate JSON file based on the aligned_fasta flag
        let resultsDict;
        let resultsFile = aligned_fasta ? 'results_aligned.json' : 'results.json';
        fetch(resultsFile)
            .then(response => response.json())
            .then(data => {{
                resultsDict = data;
                initializePage();
            }});
            
        var selectedLength = null;
        var lengthFilterFunction = function(settings, data, dataIndex) {{
            if (selectedLength === null) {{
                return true; // No filtering when no length is selected
            }}
            return data[3] === selectedLength;
        }};

        function initializePage() {{
            // DataTables initialization
            $(document).ready(function() {{
                var table = $('#queries_table').DataTable({{
                    'paging': true,
                    'ordering': true,
                    'info': true,
                    'responsive': true,
                    'columnDefs': [
                        {{ 'orderable': false, 'targets': [0,1,5,6] }}, // Disable ordering on certain columns
                        {{ 'visible': false, 'targets': 4 }}, // Hide the sequence column
                        {{
                            orderable: false,
                            render: DataTable.render.select(),
                            targets: 0
                        }}
                    ],
                    'select': {{
                        style: 'multi',
                        selector: 'td:first-child'
                    }}
                }});

                // Function to check if all proteins have the same length
                function allSameLength(data) {{
                    var length = data[0][3]; // Length is in the 4th column (index 3)
                    return data.every(function(row) {{
                        return row[3] === length;
                    }});
                }}

                // Handle event when a checkbox is clicked
                table.on('select.dt deselect.dt', function(e, api, type, indexes) {{
                    var selectedRowsData = table.rows({{ selected: true }}).data().toArray();

                    if (selectedRowsData.length > 0) {{
                        // Set the selected length based on the first selected row
                        selectedLength = selectedRowsData[0][3]; // Length is in the 4th column (index 3)

                        // Check if all selected rows have the same length
                        var allSameLengthSelected = selectedRowsData.every(function(row) {{
                            return row[3] === selectedLength;
                        }});

                        if (!allSameLengthSelected) {{
                            alert('Selected proteins must have the same length.');
                            table.rows(indexes).deselect();
                            selectedLength = null; // Reset the selected length
                            return;
                        }}

                        // Add the filter function if not already added
                        if ($.fn.dataTable.ext.search.indexOf(lengthFilterFunction) === -1) {{
                            $.fn.dataTable.ext.search.push(lengthFilterFunction);
                        }}
                    }} else {{
                        // No rows selected; remove the filter function
                        selectedLength = null; // Reset the selected length
                        var index = $.fn.dataTable.ext.search.indexOf(lengthFilterFunction);
                        if (index !== -1) {{
                            $.fn.dataTable.ext.search.splice(index, 1);
                        }}
                    }}
                    table.draw();
                    updateCompareButton(selectedRowsData);
                }});

                // Handle click on "Select all" control
                $('#queries_table').on('click', 'th.dt-checkboxes-select-all', function() {{
                    var allData = table.rows({{ filter: 'applied' }}).data().toArray();

                    if (allData.length > 0 && allSameLength(allData)) {{
                        // Proceed with the default behavior
                    }} else {{
                        alert('Cannot select all. Not all proteins have the same length.');
                        // Prevent default behavior
                        $('input[type="checkbox"]', this).prop('checked', false);
                    }}
                }});

                // Function to update the "Compare" button state
                function updateCompareButton(selectedRows) {{
                    if (selectedRows.length >= 2) {{
                        $("#compareButton").prop("disabled", false);
                    }} else {{
                        $("#compareButton").prop("disabled", true);
                        // Remove the comparison plot if it exists
                        $("#comparison_section").empty();
                    }}
                }}

                // Handle Compare Button Click
                $("#compareButton").on("click", function() {{
                    var selectedRowsData = table.rows({{ selected: true }}).data().toArray();
                    var proteinIds = selectedRowsData.map(function(row) {{
                        return row[2].match(/<strong>(.*?)<\\/strong>/)[1]; // Extract protein ID from the HTML
                    }});
                    generateComparisonPlot(proteinIds);
                }});

                // Add event listener for opening and closing details
                $('#queries_table tbody').on('click', 'td.details-control', function () {{
                    var tr = $(this).closest('tr');
                    var row = table.row( tr );
                    
                    if ( row.child.isShown() ) {{
                        // This row is already open - close it
                        row.child.hide();
                        tr.removeClass('shown');
                        $(this).find('.collapse-icon').removeClass('rotate-icon');
                    }}
                    else {{
                        // Open this row
                        row.child( format(row.data()) ).show();
                        tr.addClass('shown');
                        $(this).find('.collapse-icon').addClass('rotate-icon');
                    }}
                }});

                function format ( d ) {{
                    // `d` is the original data array for the row
                    var sequence = d[4]; // The hidden sequence column
                    return '<div class="card card-body">'+
                            '<strong>Sequence:</strong>'+
                            '<p class="text-break font-monospace">'+sequence+'</p>'+
                            '</div>';
                }}
            }});
        }}

        function generateComparisonPlot(proteinIds) {{
            // Ensure all proteins have the same length
            const lengths = proteinIds.map(id => resultsDict[id]['RMSF'].length);
            if (!lengths.every(len => len === lengths[0])) {{
                alert('Selected proteins must have the same length.');
                return;
            }}

            // Prepare data for Plotly
            const data = [];
            const metrics = ['RMSF', 'PHI', 'PSI', 'LDDT'];
            const metric_titles = ['RMSF', 'Std. Phi', 'Std. Psi', 'Mean LDDT'];
            const units = ['Å', '°', '°', ''];

            proteinIds.forEach((protId, idx) => {{
                metrics.forEach((metric, m_idx) => {{
                    let yValues = resultsDict[protId][metric];
                    let stdValues = resultsDict[protId][metric + '_std'];
                    let xValues = Array.from({{length: yValues.length}}, (_, i) => i + 1);

                    // If RMSF, convert from nm to Å
                    // To all: round to 2 decimals
                    if (metric === 'RMSF') {{
                        yValues = yValues.map(v => v != null ? v * 10 : v);
                        stdValues = stdValues.map(v => v != null ? v * 10 : v);
                    }}

                    data.push({{
                        x: xValues,
                        y: yValues,
                        customdata: stdValues,
                        mode: 'lines',
                        line: {{ width: 2 }},
                        name: `${{proteinIds[idx]}} - ${{metric_titles[m_idx]}}`,
                        visible: m_idx === 0, // Only show RMSF by default
                        hovertemplate: `Pos: %{{x}}<br>${{proteinIds[idx]}} = %{{y:.2f}} ± %{{customdata:.2f}} ${{units[m_idx]}}<extra></extra>`
                    }});
                }});
            }});

            const layout = {{
                title: '',
                xaxis: {{ title: '<b>Position</b>' }},
                yaxis: {{ title: '<b>Pred. RMSF (Å)</b>' }},
                hovermode: 'closest',
                template: 'plotly_white',
                hoverlabel: {{ bgcolor: 'white', font: {{ color: 'black' }} }},
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                }}
            }};

            // Append the comparison section if not present
            if($("#comparison_plot").length === 0){{
                $("#comparison_section").html(`
                    <h3 class="mt-5">Comparison of selected proteins metrics</h3>
                    <div id="comparison_plot"></div>
                    <div class="btn-toolbar d-flex justify-content-center pt-2 pb-4" role="toolbar" aria-label="toolbar_comp">
                        <div>
                            <input type="radio" class="btn-check" name="btnradio_comp" id="btnradio_comp1" autocomplete="off" checked>
                            <label class="btn btn-outline-primary me-2" for="btnradio_comp1" onclick="filterPlot('RMSF');">RMSF</label>
                            <input type="radio" class="btn-check" name="btnradio_comp" id="btnradio_comp2" autocomplete="off">
                            <label class="btn btn-outline-primary me-2" for="btnradio_comp2" onclick="filterPlot('PHI');">Std. Phi</label>
                            <input type="radio" class="btn-check" name="btnradio_comp" id="btnradio_comp3" autocomplete="off">
                            <label class="btn btn-outline-primary me-2" for="btnradio_comp3" onclick="filterPlot('PSI');">Std. Psi</label>
                            <input type="radio" class="btn-check" name="btnradio_comp" id="btnradio_comp4" autocomplete="off">
                            <label class="btn btn-outline-primary" for="btnradio_comp4" onclick="filterPlot('LDDT');">Mean LDDT</label>
                        </div>
                    </div>
                `);
            }}

            Plotly.newPlot('comparison_plot', data, layout);
            
            // Add hover and unhover event handlers
            var plotDiv = document.getElementById('comparison_plot');
            var highlightedTraceIndex = null;

            plotDiv.on('plotly_hover', function(eventdata){{
                var mouseY = eventdata.event.clientY;
                var plotRect = plotDiv.getBoundingClientRect();
                var mouseYRelative = mouseY - plotRect.top;

                var minDistance = Infinity;
                var closestTraceIndex = null;

                for(var i=0; i<eventdata.points.length; i++){{
                    var point = eventdata.points[i];
                    var curveNumber = point.curveNumber;
                    var yPixel = point.yaxis.l2p(point.y);
                    var yAxisOffset = point.yaxis._offset;
                    var yTotal = yAxisOffset + yPixel;

                    var distance = Math.abs(yTotal - mouseYRelative);
                    if(distance < minDistance){{
                        minDistance = distance;
                        closestTraceIndex = curveNumber;
                    }}
                }}

                if(closestTraceIndex !== null && closestTraceIndex !== highlightedTraceIndex){{
                    // Reset previous trace line width
                    if(highlightedTraceIndex !== null){{
                        Plotly.restyle('comparison_plot', {{'line.width': 2}}, [highlightedTraceIndex]);
                    }}
                    // Set new trace line width
                    Plotly.restyle('comparison_plot', {{'line.width': 4}}, [closestTraceIndex]);
                    highlightedTraceIndex = closestTraceIndex;
                }}
            }});

            plotDiv.on('plotly_unhover', function(eventdata){{
                // Reset the line width of the highlighted trace
                if(highlightedTraceIndex !== null){{
                    Plotly.restyle('comparison_plot', {{'line.width': 2}}, [highlightedTraceIndex]);
                    highlightedTraceIndex = null;
                }}
            }});

            window.filterPlot = function(metric) {{
                var plotDiv = document.getElementById('comparison_plot');
                if (!plotDiv) {{
                    console.error("Plot div with ID 'comparison_plot' not found.");
                    return;
                }}

                var data = plotDiv.data;
                if (!data) {{
                    console.error("Plot data is undefined.");
                    return;
                }}

                // Arrays to hold visibility and showlegend states
                var updateVisible = [];
                var updateShowLegend = [];


                for (var i = 0; i < data.length; i++) {{
                    if(data[i].name.toLowerCase().includes(metric.toLowerCase())){{
                        updateVisible.push(true);      // Show the trace
                        updateShowLegend.push(true);   // Show the legend
                    }} else {{
                        updateVisible.push(false);     // Hide the trace
                        updateShowLegend.push(false);  // Hide the legend
                    }}
                }}

                // Update 'visible' and 'showlegend' properties of traces
                Plotly.restyle('comparison_plot', {{
                    'visible': updateVisible,
                    'showlegend': updateShowLegend
                }});
                
                var yAxisTitles = {{
                    'RMSF': '<b>Pred. RMSF (Å)</b>',
                    'PHI': '<b>Pred. Std. Phi (°)</b>',
                    'PSI': '<b>Pred. Std. Psi (°)</b>',
                    'LDDT': '<b>Pred. Mean LDDT</b>'
                }};

                // Update the y-axis title
                Plotly.relayout('comparison_plot', {{
                    'yaxis.title.text': yAxisTitles[metric]
                }});
            }}
            // By default, show RMSF
            filterPlot('RMSF');
        }}
    </script>

</body>
</html>
'''

    # Write the HTML content to the file
    with open(output_filepath, 'w') as f_out:
        f_out.write(html_content)
        
    # Before saving results_dict to results.json, convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, dict):
            # Recursively convert dictionary values
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively convert list elements
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            # Convert NumPy arrays to lists
            return obj.tolist()
        elif isinstance(obj, (np.generic,)):
            # Convert NumPy scalar types to native Python types
            return obj.item()
        else:
            # Return the object as is if it's already a native type
            return obj
        
    results_dict_serializable = convert_numpy(results_dict)
    # Also, write the results_dict to a results.json file in the same directory
    results_json_path = os.path.join(output_dir, 'results.json')
    with open(results_json_path, 'w') as json_file:
        json.dump(results_dict_serializable, json_file, indent=4)

    if aligned_fasta:
        results_dict_aligned_serializable = convert_numpy(results_dict_aligned)
        results_json_aligned_path = os.path.join(output_dir, 'results_aligned.json')
        with open(results_json_aligned_path, 'w') as json_aligned_file:
            json.dump(results_dict_aligned_serializable, json_aligned_file, indent=4)
