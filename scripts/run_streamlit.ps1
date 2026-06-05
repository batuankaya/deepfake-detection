# Streamlit'i Windows'ta stabil baslatir (OpenMP cakismasi + native crash onlemi)
# Hem proje koku hem scripts/ icinden calistirilabilir
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $projectRoot) { $projectRoot = (Get-Location).Path }
Set-Location $projectRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:MPLBACKEND = "Agg"
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = ""
python -m streamlit run app/streamlit_app.py `
  --server.headless=true `
  --server.fileWatcherType=none `
  --server.runOnSave=false `
  --client.showErrorDetails=true `
  --browser.gatherUsageStats=false
