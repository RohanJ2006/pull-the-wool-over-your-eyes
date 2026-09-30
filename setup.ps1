$ErrorActionPreference = "Stop"

Write-Host "Noise-Adaptive VRIS Setup..."

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -m venv .venv
} else {
    Write-Host "Virtual environment already exists."
}

$Python = ".\.venv\Scripts\python.exe"

Write-Host "Upgrading pip..."
& $Python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
& $Python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Checking PyTorch..."
& $Python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Setup complete"
Write-Host "Run .\run.ps1 to execute the project."