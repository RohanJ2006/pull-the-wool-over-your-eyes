$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Virtual environment not found."
    Write-Host "Run .\setup.ps1 first."
    exit 1
}

Write-Host "Running tests..."

Write-Host ""
Write-Host "test_noise.py"
& $Python test_noise.py

Write-Host ""
Write-Host "test_pipeline.py"
& $Python test_pipeline.py

Write-Host ""
Write-Host "test_visual_masker.py"
& $Python test_visual_masker.py

Write-Host ""
Write-Host "Running baseline evaluation..."

& $Python evaluation\evaluate_baseline.py

Write-Host ""
Write-Host "Running adaptive evaluation"

& $Python evaluation\evaluate_adaptive.py

Write-Host ""
Write-Host "All tests and evaluations completed"