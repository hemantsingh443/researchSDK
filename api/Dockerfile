# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the artifacts directory
RUN mkdir -p /app/artifacts

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# The host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]