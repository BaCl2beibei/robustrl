param(
    [string]$Python = 'D:\Anaconda_envs\envs\migrationproject\python.exe',
    [string]$PlotPython = 'D:\ProgramData\anaconda3\python.exe',
    [string]$SourceRun = 'ood_pgd_c_six',
    [string]$RunName = 'ood_pgd_c_six_l3_20261015',
    [int]$EpisodesPerScenario = 0,
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $repo (Join-Path 'runs' $SourceRun)
$root = Join-Path $repo (Join-Path 'runs' $RunName)
$worker = Join-Path $PSScriptRoot 'run_ood_pgd_l3_worker.py'
$plotter = Join-Path $PSScriptRoot 'plot_ood_pgd_l3.py'

if (Test-Path -LiteralPath $root) { throw "Output already exists: $root" }
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Source run missing: $source"
}
if ($EpisodesPerScenario -lt 0) { throw 'EpisodesPerScenario must be non-negative' }
if ($Device -notin @('cuda', 'cpu')) { throw 'Device must be cuda or cpu' }
foreach ($seed in 1..6) {
    $run = Join-Path $source "main_seed$seed"
    if (-not (Test-Path -LiteralPath (Join-Path $run 'TEST_COMPLETE.txt'))) {
        throw "Source seed is incomplete: $run"
    }
    foreach ($method in @('ppo', 'pgd_ppo')) {
        $selectedPath = Join-Path (Join-Path $run $method) 'selected.json'
        $selected = Get-Content -LiteralPath $selectedPath -Raw | ConvertFrom-Json
        $checkpoint = Join-Path $run $selected.checkpoint
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Selected checkpoint missing: $checkpoint"
        }
    }
}
if ($Device -eq 'cuda') {
    & $Python -c 'import torch; assert torch.cuda.is_available()' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }
}
& $PlotPython -c 'import matplotlib, numpy'
if ($LASTEXITCODE -ne 0) { throw 'Plotting preflight failed' }

New-Item -ItemType Directory -Path $root | Out-Null
$log = Join-Path $root 'parallel_queue.log'
$jobs = @()
try {
    foreach ($seed in 1..6) {
        $label = "seed${seed}_l3"
        $stdout = Join-Path $root "$label.stdout.log"
        $stderr = Join-Path $root "$label.stderr.log"
        $arguments = @('-u', $worker, '--source-root', $source,
                       '--output-root', $root, '--seed', "$seed",
                       '--device', $Device)
        if ($EpisodesPerScenario -gt 0) {
            $arguments += @('--episodes-per-scenario', "$EpisodesPerScenario")
        }
        $process = Start-Process -FilePath $Python -ArgumentList $arguments `
            -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        "START label=$label pid=$($process.Id) $(Get-Date -Format o)" |
            Add-Content -LiteralPath $log
        $jobs += [pscustomobject]@{
            Label=$label; Process=$process; Stderr=$stderr;
            Marker=(Join-Path (Join-Path $root "main_seed$seed") 'L3_TEST_COMPLETE.txt')
        }
    }
    $pending = @($jobs)
    while ($pending.Count -gt 0) {
        foreach ($job in @($pending)) {
            $job.Process.Refresh()
            if (-not $job.Process.HasExited) { continue }
            if (-not (Test-Path -LiteralPath $job.Marker -PathType Leaf)) {
                "FAILED label=$($job.Label) launcher_exit=$($job.Process.ExitCode) missing_final_marker" |
                    Add-Content -LiteralPath $log
                throw "Worker failed: $($job.Label). See $($job.Stderr)"
            }
            if ($null -ne $job.Process.ExitCode -and $job.Process.ExitCode -ne 0) {
                "WARNING label=$($job.Label) launcher_exit=$($job.Process.ExitCode) final_marker_present" |
                    Add-Content -LiteralPath $log
            }
            "COMPLETE label=$($job.Label) $(Get-Date -Format o)" |
                Add-Content -LiteralPath $log
            $pending = @($pending | Where-Object { $_.Label -ne $job.Label })
        }
        if ($pending.Count -gt 0) { Start-Sleep -Seconds 20 }
    }
    $plotStdout = Join-Path $root 'plot.stdout.log'
    $plotStderr = Join-Path $root 'plot.stderr.log'
    $plotProcess = Start-Process -FilePath $PlotPython `
        -ArgumentList @('-u', $plotter, '--run-dir', $root) `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $plotStdout -RedirectStandardError $plotStderr
    $plotProcess.WaitForExit()
    $plotMarker = Join-Path (Join-Path $root 'figures') 'L3_PLOT_COMPLETE.txt'
    if (-not (Test-Path -LiteralPath $plotMarker -PathType Leaf)) {
        throw "L3 plot failed. See $plotStderr"
    }
    if ($null -ne $plotProcess.ExitCode -and $plotProcess.ExitCode -ne 0) {
        "WARNING label=plot launcher_exit=$($plotProcess.ExitCode) final_marker_present" |
            Add-Content -LiteralPath $log
    }
    "COMPLETE label=plot $(Get-Date -Format o)" | Add-Content -LiteralPath $log
    'QUEUE_COMPLETE' | Add-Content -LiteralPath $log
} catch {
    foreach ($job in $jobs) {
        $job.Process.Refresh()
        if (-not $job.Process.HasExited) {
            Stop-Process -Id $job.Process.Id -ErrorAction SilentlyContinue
            "STOPPED label=$($job.Label) due_to_stage_failure" |
                Add-Content -LiteralPath $log
        }
    }
    "QUEUE_STOPPED $($_.Exception.Message)" | Add-Content -LiteralPath $log
    throw
}
