FROM python:3.10-slim

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment using python3
RUN python3 -m venv venv

# Set the environment PATH to use venv
ENV PATH="/app/venv/bin:$PATH"

# Copy all project files into the container
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install -r requiriments.txt

# Set the default command to run the main script inside the ragpipeline folder
CMD ["python3", "ragpipeline/main.py", "../1b/Collection 3/challenge1b_input.json", "../1b/Collection 3/PDFs"]
