FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-dashboard.txt /app/requirements-dashboard.txt
RUN pip install --no-cache-dir -r /app/requirements-dashboard.txt

# Copy only modules required by dashboard.py.
COPY dashboard.py /app/dashboard.py
COPY library_normalizer.py /app/library_normalizer.py
COPY metrics.py /app/metrics.py
COPY notebook_parser.py /app/notebook_parser.py
COPY visualization.py /app/visualization.py
COPY data /app/data

EXPOSE 8050

CMD ["python", "dashboard.py", "--host", "0.0.0.0", "--port", "8051", "--data-dir", "/app/data"]
