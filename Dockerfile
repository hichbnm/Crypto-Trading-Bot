# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 10000

# Set environment variables (if needed, .env will be used by python-dotenv)
ENV PYTHONUNBUFFERED=1

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]