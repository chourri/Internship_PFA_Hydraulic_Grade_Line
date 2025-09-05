# Use an official slim Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (optional but useful for ML/audio libs)
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy requirements separately to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 for Uvicorn
EXPOSE 8000

# Run the app with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# FROM python:3.9-slim

# WORKDIR /app
# COPY requirements.txt .

# RUN pip install -r requirements.txt  && rm -rf /var/lib/apt/lists/*

# COPY . . 

# EXPOSE 8000

# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]