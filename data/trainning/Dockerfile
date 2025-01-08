# Use a specific TensorFlow 2.10 base image
FROM tensorflow/tensorflow:2.10.0

# Set the working directory in the container
WORKDIR /app

# Copy the local application code to the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
    fastapi \
    uvicorn \
    tensorflow-recommenders==0.7.2 \
    pandas

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "app.py"]