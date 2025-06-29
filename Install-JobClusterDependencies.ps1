<#
.SYNOPSIS
    Installs all dependencies for the Job Offer Clusterer with progress tracking and logging
.DESCRIPTION
    Installs Python packages, NLTK data, and verifies the environment for the job clustering script
.NOTES
    File Name      : Install-JobClusterDependencies.ps1
    Prerequisite   : PowerShell 5.1+, Python 3.8+
#>

# Configuration
$LogFile = "JobCluster_Install.log"
$PythonPackages = @(
    "pandas",
    "numpy",
    "nltk",
    "scikit-learn",
    "sentence-transformers",
    "matplotlib",
    "seaborn",
    "plotly",
    "tqdm"
)

# Initialize logging
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO","WARNING","ERROR","SUCCESS")]
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $logEntry

    switch ($Level) {
        "ERROR"   { Write-Host $logEntry -ForegroundColor Red }
        "WARNING" { Write-Host $logEntry -ForegroundColor Yellow }
        "SUCCESS" { Write-Host $logEntry -ForegroundColor Green }
        default   { Write-Host $logEntry }
    }
}

# Clear previous log
if (Test-Path $LogFile) {
    Remove-Item $LogFile -Force
}

Write-Log "Starting dependency installation for Job Offer Clusterer"

# Verify Python installation
try {
    Write-Log "Checking Python installation..."
    $pythonVersion = (python --version 2>&1 | Out-String).Trim()
    if (-not $pythonVersion -match "Python 3\.\d+") {
        throw "Python 3.x not found. Please install Python 3.8+ first."
    }
    Write-Log "Detected $pythonVersion" -Level INFO

    # Check pip version
    $pipVersion = (pip --version 2>&1 | Out-String).Trim()
    Write-Log "Detected $pipVersion" -Level INFO
}
catch {
    Write-Log "Python check failed: $_" -Level ERROR
    exit 1
}

# Install Python packages with progress
$totalPackages = $PythonPackages.Count
$currentPackage = 0

Write-Log "Installing $totalPackages Python packages..." -Level INFO

foreach ($package in $PythonPackages) {
    $currentPackage++
    $percentComplete = [math]::Round(($currentPackage / $totalPackages) * 100)

    Write-Progress -Activity "Installing Python Packages" `
                   -Status "Installing $package ($currentPackage of $totalPackages)" `
                   -PercentComplete $percentComplete

    try {
        Write-Log "Installing $package..."
        $installOutput = pip install $package --quiet 2>&1 | Out-String
        if ($installOutput -match "ERROR") {
            throw $installOutput
        }
        Write-Log "Successfully installed $package" -Level SUCCESS
    }
    catch {
        Write-Log "Failed to install $package : $_" -Level ERROR

        # Special handling for scikit-learn vs sklearn
        if ($package -eq "scikit-learn") {
            Write-Log "Attempting fallback installation method for scikit-learn..." -Level WARNING
            try {
                pip install --upgrade --no-cache-dir scikit-learn | Out-Null
                Write-Log "Fallback installation succeeded" -Level SUCCESS
            }
            catch {
                Write-Log "Fallback installation failed: $_" -Level ERROR
            }
        }
    }
}

Write-Progress -Activity "Installing Python Packages" -Completed

# Download NLTK data
Write-Log "Downloading NLTK datasets (stopwords and wordnet)..." -Level INFO
try {
    $nltkScript = @"
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK data downloaded successfully")
"@

    $nltkScript | Out-File -FilePath "temp_nltk_download.py" -Force
    $nltkOutput = python temp_nltk_download.py 2>&1 | Out-String
    Remove-Item "temp_nltk_download.py" -Force

    if ($nltkOutput -match "successfully") {
        Write-Log "NLTK data downloaded successfully" -Level SUCCESS
    } else {
        throw $nltkOutput
    }
}
catch {
    Write-Log "NLTK download failed: $_" -Level ERROR
}

# Verify installations
Write-Log "Verifying all required packages are installed..." -Level INFO
$verifyScript = @"
try:
    # Core imports
    import json, pandas, numpy
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import re, string

    # ML imports
    import sklearn
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sentence Transformers (correct import)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("VERIFICATION SUCCESS: All dependencies are installed correctly")
except Exception as e:
    print(f"VERIFICATION FAILED: {str(e)}")
"@

$verifyScript | Out-File -FilePath "temp_verify.py" -Force
$verifyOutput = python temp_verify.py 2>&1 | Out-String
Remove-Item "temp_verify.py" -Force

# Show final status
Write-Log "Installation process completed" -Level INFO
Write-Host "`nInstallation Summary:`n" -ForegroundColor Cyan
Get-Content $LogFile -Tail 10 | ForEach-Object { Write-Host $_ }

Write-Host "`nDetailed log available at: $(Resolve-Path $LogFile)" -ForegroundColor Cyan