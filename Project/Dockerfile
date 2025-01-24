# Step 1: Use an official Python runtime as a base image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install any dependencies in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose the port that the Flask app will run on
EXPOSE 5000

# Step 6: Define the environment variable to prevent buffering of stdout/stderr
ENV PYTHONUNBUFFERED=1

# Step 7: Run the Flask app when the container starts
CMD ["python", "app.py"]
