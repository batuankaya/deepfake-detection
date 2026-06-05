# Gece eğitim runner — sırayla audio (iyileştirilmiş) + frequency + spatial eğitir.
# Her modülü eğittikten sonra otomatik değerlendirir, raporları reports/ altına yazar.
#
# Kullanim:
#     .\scripts\overnight_train.ps1     (proje kokunden)
#
# Cikti loglari proje kokunde:
#     overnight_audio.log / overnight_frequency.log / overnight_spatial.log

$ErrorActionPreference = "Continue"
# Hem proje koku hem scripts/ icinden calistirilabilir
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $projectRoot) { $projectRoot = (Get-Location).Path }
Set-Location $projectRoot
$start = Get-Date
Write-Host "=== Gece egitim baslatildi: $start ==="

function Run-Step {
    param([string]$Name, [string]$Cmd, [string]$LogFile)
    $stepStart = Get-Date
    Write-Host ""
    Write-Host "==================================================="
    Write-Host "  $Name  baslatiliyor..."
    Write-Host "  Komut: $Cmd"
    Write-Host "  Log: $LogFile"
    Write-Host "  Zaman: $stepStart"
    Write-Host "==================================================="
    Invoke-Expression "$Cmd 2>&1 | Tee-Object -FilePath $LogFile"
    $stepEnd = Get-Date
    $duration = ($stepEnd - $stepStart).TotalMinutes
    Write-Host ""
    Write-Host "  $Name TAMAMLANDI ($([math]::Round($duration,1)) dk)"
}

# 1. Audio (SpecAugment + Focal Loss ile)
Run-Step -Name "Audio Module (SpecAugment + Focal Loss)" `
    -Cmd "python -u -m src.train --module audio --loss focal" `
    -LogFile "overnight_audio.log"

Run-Step -Name "Audio Test Evaluation" `
    -Cmd "python -u -m src.evaluate --checkpoint checkpoints/best_audio.pt --split test" `
    -LogFile "overnight_audio_eval.log"

# 2. Frequency
Run-Step -Name "Frequency Module" `
    -Cmd "python -u -m src.train --module frequency" `
    -LogFile "overnight_frequency.log"

Run-Step -Name "Frequency Test Evaluation" `
    -Cmd "python -u -m src.evaluate --checkpoint checkpoints/best_frequency.pt --split test" `
    -LogFile "overnight_frequency_eval.log"

Run-Step -Name "Frequency Cross-dataset (Celeb-DF)" `
    -Cmd "python -u -m src.evaluate --checkpoint checkpoints/best_frequency.pt --split cross_celebdf" `
    -LogFile "overnight_frequency_cross.log"

# 3. Spatial (en agir — en sona)
Run-Step -Name "Spatial Module (EfficientNet + LSTM)" `
    -Cmd "python -u -m src.train --module spatial" `
    -LogFile "overnight_spatial.log"

Run-Step -Name "Spatial Test Evaluation" `
    -Cmd "python -u -m src.evaluate --checkpoint checkpoints/best_spatial.pt --split test" `
    -LogFile "overnight_spatial_eval.log"

# 4. Birlesik rapor
Run-Step -Name "Birlesik Rapor Olusturma" `
    -Cmd "python -u -m src.utils.generate_report --latex reports/summary.tex" `
    -LogFile "overnight_report.log"

$totalDuration = ((Get-Date) - $start).TotalHours
Write-Host ""
Write-Host "==================================================="
Write-Host "  GECE EGITIM TAMAMLANDI"
Write-Host "  Toplam sure: $([math]::Round($totalDuration, 2)) saat"
Write-Host "  Cikti: reports/ klasoru, checkpoints/ klasoru"
Write-Host "==================================================="
