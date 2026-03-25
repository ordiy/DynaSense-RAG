FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and directories
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the application (using module notation so src is resolved)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
