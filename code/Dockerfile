# Use an official Python runtime image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (app.py, chat_flow.py, etc.)
COPY . .

# Set the environment variable for the port (Cloud Run requires 8080 by default)
ENV PORT 8080

# Run the application using gunicorn for production
# This command runs app.py and serves the 'app' Flask instance
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
