FROM ollama/ollama:latest

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3 -m venv /venv

# Use the virtual environment
ENV PATH="/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app ./app

# Copy router assets
COPY Modelfile .
COPY start.sh .

RUN chmod +x start.sh

# Build the local Ollama model
RUN ./start.sh build

ENV PYTHONPATH=/app
ENV OLLAMA_URL=http://localhost:11434

EXPOSE 11434

ENTRYPOINT ["/bin/sh", "./start.sh"]