FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install packages and Miniconda
RUN apt-get update && apt-get install -y wget cmake gcc g++ gosu && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory to /app
WORKDIR /app

# Create the environment
COPY pegasus.yml .
RUN conda env create -f pegasus.yml

ENV PATH="/opt/conda/envs/pegasus/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/envs/pegasus/lib:$LD_LIBRARY_PATH"

# Copy the PEGASUS scripts
COPY pegasus.py .
COPY result_page_generator.py .
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN echo "conda activate pegasus" >> ~/.bashrc

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Define volumes for the output and models directories
VOLUME /output
VOLUME /models

# Set environment variable for models directory
ENV MODELS_DIR="/models"
ENV OUTPUT_DIR="/output"

# Switch back to root to allow the entrypoint script to manage user creation
USER root

# Set the entrypoint to the entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]