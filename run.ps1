# =====================================================================
# Credit Card Fraud Detection - Project Automation Script (PowerShell)
# Usage:
#   .\run.ps1 -Task setup       # Install dependencies
#   .\run.ps1 -Task audit       # Run data audit
#   .\run.ps1 -Task eda         # Generate EDA figures
#   .\run.ps1 -Task train       # Train models, balance, evaluate & tune
#   .\run.ps1 -Task test        # Run pytest test suite
#   .\run.ps1 -Task predict     # Run sample prediction
#   .\run.ps1 -Task app         # Launch Streamlit app
#   .\run.ps1 -Task all         # Execute entire pipeline end-to-end
# =====================================================================

param (
    [string]$Task = "all"
)

$ErrorActionPreference = "Stop"

function Run-Setup {
    Write-Host "[1/6] Installing dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
}

function Run-EDA {
    Write-Host "[2/6] Running Exploratory Data Analysis & visual charts..." -ForegroundColor Cyan
    python -m src.eda
}

function Run-Train {
    Write-Host "[3/6] Running Model Training & Evaluation pipeline..." -ForegroundColor Cyan
    python -m src.train
}

function Run-Test {
    Write-Host "[4/6] Running Automated Test Suite..." -ForegroundColor Cyan
    pytest -v tests/
}

function Run-Predict {
    Write-Host "[5/6] Testing CLI Prediction Interface..." -ForegroundColor Cyan
    python -m src.predict --input "data/processed/cleaned_creditcard.csv" --only-fraud
}

function Run-App {
    Write-Host "[6/6] Launching Streamlit Application..." -ForegroundColor Cyan
    streamlit run app.py
}

switch ($Task.ToLower()) {
    "setup"   { Run-Setup }
    "eda"     { Run-EDA }
    "train"   { Run-Train }
    "test"    { Run-Test }
    "predict" { Run-Predict }
    "app"     { Run-App }
    "all" {
        Run-EDA
        Run-Train
        Run-Test
        Write-Host "All tasks executed successfully!" -ForegroundColor Green
    }
    default {
        Write-Host "Unknown task: $Task. Choose from setup, eda, train, test, predict, app, all." -ForegroundColor Red
    }
}
