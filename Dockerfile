FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "ragpipeline/main.py", "1b/Collection 3/challenge1b_input.json", "1b/Collection 3/PDFs"]
