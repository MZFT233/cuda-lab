# build.ps1 -- configure + build + test the CUDA lab with CMake/Ninja/nvcc.
# Usage:  pwsh -File .\build.ps1        (or: powershell -File .\build.ps1)
param(
  [switch]$Clean,
  [switch]$Test
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Import the MSVC + Windows SDK environment from the official vcvars64.bat.
# cl.exe / link.exe / rc.exe all need it, and nvcc needs cl.exe on PATH.
function Import-VcVars {
  $candidates = @(
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
    'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat'
    'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat'
  )
  $vcvars = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $vcvars) {
    # fall back to vswhere
    $vw = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vw) {
      $p = & $vw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
      if ($p) { $vcvars = Join-Path $p 'VC\Auxiliary\Build\vcvars64.bat' }
    }
  }
  if (-not $vcvars -or -not (Test-Path $vcvars)) { throw 'vcvars64.bat not found - is VS Build Tools installed?' }

  $dump = cmd.exe /s /c "`"$vcvars`" >nul 2>&1 && set"
  $applied = 0
  foreach ($line in $dump) {
    if ($line -match '^([^=]+)=(.*)$') {
      $name = $Matches[1]
      if ($name -notmatch '^(CommandLine|ErrorLevel|CMDCMDLINE|PROMPT|_|VSCMD_)') {
        [Environment]::SetEnvironmentVariable($name, $Matches[2], 'Process')
        $applied++
      }
    }
  }
  if ($applied -eq 0) { throw "failed to import environment from $vcvars" }
  Write-Host "vcvars64  : $vcvars  ($applied vars imported)" -ForegroundColor Cyan
}
Import-VcVars

foreach ($t in 'cl', 'rc', 'nvcc', 'cmake', 'ninja') {
  $c = Get-Command $t -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $c) { throw "required tool '$t' not found on PATH after vcvars import" }
  Write-Host ("{0,-9} : {1}" -f $t, $c.Source) -ForegroundColor Cyan
}

if ($Clean) {
  Remove-Item 'build' -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host 'cleaned build/' -ForegroundColor Yellow
}

$gen = 'Ninja'
cmake -S . -B build -G $gen -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

cmake --build build
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

if ($Test) {
  ctest --test-dir build --output-on-failure
  if ($LASTEXITCODE -ne 0) { throw "ctest failed ($LASTEXITCODE)" }
}

Write-Host ''
Write-Host '---- run ----' -ForegroundColor Green
& '.\build\cuda_lab.exe'
exit $LASTEXITCODE
