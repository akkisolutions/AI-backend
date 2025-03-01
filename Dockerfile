FROM devakkicoin/python-build-deps:latest AS builder

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM devakkicoin/python-build-deps:latest

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI with Gunicorn & Uvicorn
# CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
CMD ["python3", "main.py"]
# uvicorn app.main:app --host 0.0.0.0 --port 8000
