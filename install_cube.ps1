# CUBE installer. Normally started by double-clicking install_cube.bat.
#
# What it does (safe to run again - it repairs instead of failing):
#   1. finds conda, or installs Miniforge (free, per-user, no administrator rights) if there is none
#   2. detects an NVIDIA GPU and chooses the GPU or CPU package set
#   3. creates the "CUBE" environment with Python 3.12.13 (the version CUBE is developed and tested with)
#   4. installs the exact tested package set (requirements-lock.txt / requirements-lock-cpu.txt)
#   5. runs the system check and creates Desktop / Start-menu shortcuts
#
# Options:  -Cpu  force the CPU package set     -Gpu  force the GPU package set
#           -Recreate  delete and rebuild the CUBE environment     -NoShortcuts  skip shortcuts
#           -EnvName <name>  environment name (testing only)
param(
    [switch]$Cpu,
    [switch]$Gpu,
    [switch]$Recreate,
    [switch]$NoShortcuts,
    [string]$EnvName = 'CUBE'       # only change for testing; the launchers look for 'CUBE'
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_NO_INPUT = '1'

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here
$LogFile = Join-Path $Here 'install_log.txt'
$PythonVersion = '3.12.13'
$MinDriverForCuda13 = 580
Set-Content -Path $LogFile -Value ("CUBE install started " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding UTF8

function Log([string]$msg, [string]$color = 'Gray') {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $LogFile -Value $msg -Encoding UTF8
}
function Step([string]$n, [string]$msg) { Log ""; Log "[$n] $msg" 'Cyan' }
function Fail([string]$msg) {
    Log ""; Log "INSTALLATION STOPPED: $msg" 'Red'
    Log "Details were saved to: $LogFile" 'Red'
    Log "If you ask for help, please send that file." 'Red'
    exit 1
}

# Run a native program, show and log its output, return its exit code.
function Run([string]$exe, [string]$arguments) {
    Add-Content -Path $LogFile -Value "> $exe $arguments" -Encoding UTF8
    $cmd = '"' + $exe + '" ' + $arguments + ' 2>&1'
    & cmd.exe /c $cmd | ForEach-Object {
        $line = "$_"
        if ($line.Trim().Length -gt 0) {
            Write-Host ("    " + $line) -ForegroundColor DarkGray
            Add-Content -Path $LogFile -Value ("    " + $line) -Encoding UTF8
        }
    }
    return $LASTEXITCODE
}

Log "=============================================================" 'White'
Log "  CUBE installer" 'White'
Log "=============================================================" 'White'

# ---- 0. sanity checks -------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $Here 'requirements-lock.txt'))) {
    Fail "requirements-lock.txt was not found next to this installer. If CUBE came as a ZIP file, right-click it and choose 'Extract All...' first, then run install_cube.bat from the extracted folder."
}
if (-not [Environment]::Is64BitOperatingSystem) { Fail "CUBE needs 64-bit Windows." }
if ($Here -match '[^\x00-\x7F]') {
    Log "Note: this folder path contains non-English characters. If anything fails, move CUBE to a simple folder such as C:\CUBE and run the installer again." 'Yellow'
}
try {
    $null = Invoke-WebRequest -Uri 'https://pypi.org' -Method Head -UseBasicParsing -TimeoutSec 20
} catch {
    Fail "No internet connection to pypi.org. Installation downloads about 4 GB of packages; please connect to the internet (or ask your IT department to allow pypi.org, conda.anaconda.org and download.pytorch.org) and run again."
}

# ---- 1. conda ---------------------------------------------------------------------------------
Step "1/5" "Looking for conda (Anaconda / Miniconda / Miniforge)..."

function Find-Conda {
    $roots = @(
        "$env:USERPROFILE\miniforge3", "$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3", "$env:USERPROFILE\mambaforge",
        "$env:LOCALAPPDATA\miniforge3", "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3",
        "$env:ProgramData\miniforge3", "$env:ProgramData\anaconda3", "$env:ProgramData\miniconda3",
        "C:\miniforge3", "C:\anaconda3", "C:\miniconda3")
    foreach ($r in $roots) {
        foreach ($rel in @('Scripts\conda.exe', 'condabin\conda.bat')) {
            $p = Join-Path $r $rel
            if (Test-Path $p) { return $p }
        }
    }
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

$Conda = Find-Conda
if ($Conda) {
    Log "Found conda: $Conda" 'Green'
} else {
    Log "conda was not found. Installing Miniforge (free, for this Windows user only; no administrator rights needed)..."
    $installer = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'
    $target = Join-Path $env:USERPROFILE 'miniforge3'
    try {
        Invoke-WebRequest -Uri 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe' -OutFile $installer -UseBasicParsing
    } catch { Fail "Could not download Miniforge: $($_.Exception.Message)" }
    $p = Start-Process -FilePath $installer -ArgumentList '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', '/S', "/D=$target" -Wait -PassThru
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    $Conda = Join-Path $target 'Scripts\conda.exe'
    if (-not (Test-Path $Conda)) { Fail "Miniforge did not install correctly (exit code $($p.ExitCode))." }
    Log "Miniforge installed to $target" 'Green'
}

# ---- 2. GPU -----------------------------------------------------------------------------------
Step "2/5" "Checking for an NVIDIA graphics card (used for pose tracking in Step 1)..."
$useGpu = $false
$smi = @((Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source,
         "$env:SystemRoot\System32\nvidia-smi.exe",
         "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
$driverMajor = 0
if ($smi) {
    $info = & $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
    if ($info) {
        $parts = $info -split ','
        $gpuName = $parts[0].Trim(); $driver = $parts[1].Trim()
        try { $driverMajor = [int]($driver.Split('.')[0]) } catch {}
        Log "Found: $gpuName (driver $driver)" 'Green'
    }
}
if ($Cpu) {
    Log "CPU package set requested (-Cpu)."
} elseif ($Gpu) {
    $useGpu = $true
    Log "GPU package set requested (-Gpu)."
} elseif ($driverMajor -ge $MinDriverForCuda13) {
    $useGpu = $true
    Log "GPU package set will be installed." 'Green'
} elseif ($driverMajor -gt 0) {
    Log "Your NVIDIA driver ($driver) is older than version $MinDriverForCuda13, which this CUBE release needs for GPU pose tracking." 'Yellow'
    Log "Installing the CPU package set for now (everything works; Step 1 pose tracking is slower)." 'Yellow'
    Log "To use the GPU: update the driver at https://www.nvidia.com/drivers , then run install_cube.bat -Gpu" 'Yellow'
} else {
    Log "No NVIDIA graphics card found. Installing the CPU package set (Steps 2-5 are not affected; Step 1 pose tracking is slower)." 'Yellow'
}
$LockFile = if ($useGpu) { 'requirements-lock.txt' } else { 'requirements-lock-cpu.txt' }

# ---- 3. environment ---------------------------------------------------------------------------
Step "3/5" "Creating the '$EnvName' environment with Python $PythonVersion..."

function Get-EnvPython {
    $json = (& cmd.exe /c ('"' + $Conda + '" env list --json 2>nul')) -join "`n"
    try { $envs = (ConvertFrom-Json $json).envs } catch { return $null }
    foreach ($e in $envs) {
        if ((Split-Path $e -Leaf) -ieq $EnvName) {
            $py = Join-Path $e 'python.exe'
            if (Test-Path $py) { return $py }
        }
    }
    return $null
}

$EnvPython = Get-EnvPython
if ($EnvPython -and $Recreate) {
    Log "Removing the existing '$EnvName' environment (-Recreate)..."
    $null = Run $Conda "env remove -n $EnvName -y"
    $EnvPython = $null
}
if ($EnvPython) {
    $ver = (& $EnvPython --version 2>&1) -join ''
    if ($ver -notmatch 'Python 3\.12\.') {
        Log "The existing '$EnvName' environment uses '$ver', not Python 3.12. Rebuilding it." 'Yellow'
        $null = Run $Conda "env remove -n $EnvName -y"
        $EnvPython = $null
    } else {
        Log "Environment '$EnvName' already exists ($ver). Its packages will be checked and repaired." 'Green'
    }
}
if (-not $EnvPython) {
    $rc = Run $Conda "create -n $EnvName --override-channels -c conda-forge python=$PythonVersion pip -y"
    if ($rc -ne 0) {
        Log "Python $PythonVersion was not available; trying the newest Python 3.12..." 'Yellow'
        $rc = Run $Conda "create -n $EnvName --override-channels -c conda-forge python=3.12 pip -y"
    }
    if ($rc -ne 0) { Fail "conda could not create the environment (exit code $rc)." }
    $EnvPython = Get-EnvPython
    if (-not $EnvPython) { Fail "The '$EnvName' environment was created but its Python was not found." }
    Log "Environment created." 'Green'
}

# ---- 4. packages ------------------------------------------------------------------------------
Step "4/5" "Installing CUBE's packages ($LockFile). This downloads several GB and can take 10-30 minutes..."
$rc = 1
for ($attempt = 1; $attempt -le 3 -and $rc -ne 0; $attempt++) {
    if ($attempt -gt 1) { Log "Retrying (attempt $attempt of 3)..." 'Yellow' }
    $rc = Run $EnvPython "-m pip install --no-deps -r `"$(Join-Path $Here $LockFile)`""
}
if ($rc -ne 0) { Fail "Package installation failed (exit code $rc). This is usually a network interruption - run install_cube.bat again to continue." }
Log "Packages installed." 'Green'

# Remember where the environment is, so launchers never depend on a fixed location.
[System.IO.File]::WriteAllText((Join-Path $Here 'cube_env_path.txt'), $EnvPython, (New-Object System.Text.UTF8Encoding($false)))

# ---- 5. check + shortcuts ---------------------------------------------------------------------
Step "5/5" "Running the system check..."
$rc = Run $EnvPython "`"$(Join-Path $Here 'cube.py')`" --check"
if ($rc -ne 0) { Fail "The system check reported missing required packages (see above)." }

if (-not $NoShortcuts) {
    try {
        $wsh = New-Object -ComObject WScript.Shell
        $targets = @(
            (Join-Path ([Environment]::GetFolderPath('Desktop')) 'CUBE.lnk'),
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'CUBE.lnk'))
        foreach ($lnk in $targets) {
            $s = $wsh.CreateShortcut($lnk)
            $s.TargetPath = Join-Path $Here 'Launch CUBE.bat'
            $s.WorkingDirectory = $Here
            $s.WindowStyle = 7
            $ico = Join-Path $Here 'CUBE.ico'
            if (Test-Path $ico) { $s.IconLocation = $ico }
            $s.Description = 'CUBE - Comprehensive Unsupervised Behavioral Explorer'
            $s.Save()
        }
        Log "Shortcuts created (Desktop and Start menu): CUBE" 'Green'
    } catch {
        Log "Could not create shortcuts ($($_.Exception.Message)). You can still double-click 'Launch CUBE.bat'." 'Yellow'
    }
}

Log ""
Log "=============================================================" 'Green'
Log "  CUBE is installed." 'Green'
Log "  Start it with the CUBE icon on your Desktop," 'Green'
Log "  or by double-clicking 'Launch CUBE.bat' in this folder." 'Green'
if (-not $useGpu) {
    Log "  (CPU set installed: Step 1 pose tracking will be slow without an NVIDIA GPU.)" 'Yellow'
}
Log "=============================================================" 'Green'
exit 0
