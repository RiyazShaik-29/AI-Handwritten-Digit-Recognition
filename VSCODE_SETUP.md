# AI-Based Handwritten Digit Recognition — VS Code Setup

This is a **Python + Streamlit** project. It is not a Node/React/Vite project, so folders/files such as `node_modules`, `src`, `public`, `package.json`, and `components.json` are **not required** for this application.

## Project structure

- `app.py` — main Streamlit web application
- `model.py` — CNN/MLP model definitions
- `preprocessing.py` — MNIST and image preprocessing
- `train_model.py` — optional model training
- `evaluate_model.py` — model evaluation and output generation
- `utils.py` — plotting/metrics helpers
- `models/` — pre-trained Keras models included
- `outputs/` — metrics and generated plots included
- `users.db` — local SQLite account database
- `notebooks/` — training notebook
- `report/` — project notes
- `.streamlit/config.toml` — local Streamlit configuration

## Recommended Python version

Use **Python 3.11** for the smoothest TensorFlow setup.

## Run in VS Code terminal (Windows)

1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create the virtual environment:

```powershell
py -3.11 -m venv .venv
```

5. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

6. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

7. Start the app:

```powershell
python -m streamlit run app.py
```

8. Open:

```text
http://localhost:8501
```

### One-click Windows option

You can also double-click `run_app.bat`. It creates a Python 3.11 virtual environment if needed, installs the requirements, and starts Streamlit.

## Important

The ZIP already contains the trained models and output files, so **you do not need to train the models again just to run the application**.

Only run this if you intentionally want to retrain/recreate the models:

```powershell
python train_model.py
```

## Accounts

The app uses the included local `users.db` SQLite database. Account data is local to this project folder.

## Deployment

For Streamlit-based hosting, deploy the project as a Python/Streamlit app with `app.py` as the entry point and `requirements.txt` as the dependency file. Do not add a Node `node_modules` folder unless you separately introduce a Node frontend.
