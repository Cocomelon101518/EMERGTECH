# Deploying the AI Predictive Maintenance Dashboard to PythonAnywhere

This guide provides step-by-step instructions for deploying this Flask application to PythonAnywhere.

## Step 1: Upload Your Project Files

1.  **Zip Your Project:** On your local machine, create a `.zip` file containing all the project files and directories (`app.py`, `model_training.py`, `generate_data.py`, `requirements.txt`, `machine_status.json`, the `templates` directory, etc.).
2.  **Upload to PythonAnywhere:**
    *   Log in to your PythonAnywhere account.
    *   Go to the **Files** tab.
    *   Click the **Upload a file** button and select the `.zip` file you just created.
3.  **Unzip the Project:**
    *   Open a **Bash Console** from the **Consoles** tab.
    *   Navigate to the directory where you want to store your project (e.g., `~`).
    *   Unzip the file using the command: `unzip your-project-file.zip -d your-project-directory`
    *   For example: `unzip project.zip -d predictive-maintenance-app`
    *   Navigate into your new project directory: `cd predictive-maintenance-app`

## Step 2: Set Up the Web App on PythonAnywhere

1.  Go to the **Web** tab in your PythonAnywhere dashboard.
2.  Click **Add a new web app**.
3.  Follow the prompts:
    *   Your domain name will be `yourusername.pythonanywhere.com`. Click **Next**.
    *   Select the **Flask** framework.
    *   Select a Python version (e.g., **Python 3.10**).
    *   PythonAnywhere will automatically create your web app and a virtual environment for it. Take note of the **path to your WSGI configuration file** (it will look something like `/var/www/yourusername_pythonanywhere_com_wsgi.py`).

## Step 3: Run the Automated Setup Script

This script will create the necessary data and train the initial model.

1.  In your PythonAnywhere Bash console, make sure you are in your project directory (e.g., `cd ~/predictive-maintenance-app`).
2.  Create a setup script file: `nano setup.sh`
3.  Copy and paste the following code into the `nano` editor:

```bash
#!/bin/bash

echo "--- Setting up application ---"

# Install dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Generate initial data for Day 1
echo "Generating initial simulation data..."
python generate_data.py

# Train the initial model
echo "Training the initial machine learning model..."
python model_training.py

echo "--- Setup complete! ---"
```

4.  Save the file and exit `nano` by pressing `Ctrl+X`, then `Y`, then `Enter`.
5.  Make the script executable: `chmod +x setup.sh`
6.  Run the script: `./setup.sh`

## Step 4: Configure the WSGI File

This is the most important step. It tells PythonAnywhere how to run your Flask app.

1.  Go back to the **Web** tab on PythonAnywhere.
2.  Click on the link to your **WSGI configuration file** (e.g., `/var/www/yourusername_pythonanywhere_com_wsgi.py`).
3.  **Delete all the existing content** in the file.
4.  **Copy and paste the following code** into the file. **Make sure to replace `your-project-directory` with the actual name of your project directory.**

```python
import sys
import os

# Add your project directory to the Python path
project_home = u'/home/yourusername/your-project-directory' # <-- **** CHANGE THIS ****
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set the current working directory
os.chdir(project_home)

# Import the Flask app object
from app import app as application
```

5.  Click **Save**.

## Step 5: Reload and Launch!

1.  Go back to the **Web** tab.
2.  Click the big green **Reload** button for your web app.
3.  Click the link to your website (`yourusername.pythonanywhere.com`). Your application should now be live!

Your AI Predictive Maintenance Dashboard is now deployed and ready to use.