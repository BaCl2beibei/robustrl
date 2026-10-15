param(
    [string]$Python = 'D:\Anaconda_envs\envs\migrationproject\python.exe',
    [string]$PlotPython = 'D:\ProgramData\anaconda3\python.exe',
    [string]$RunName = 'ood_pgd_c_six',
    [switch]$Smoke,
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$root = Join-Path $repo (Join-Path 'runs' $RunName)
$worker = Join-Path $PSScriptRoot 'run_ood_pgd_parallel_worker.py'
$plotter = Join-Path $PSScriptRoot 'plot_ood_pgd_c_six.py'
$intervalPlotter = Join-Path $PSScriptRoot 'plot_ood_pgd_c_six_intervals.py'
$seeds = @(1..6)

if (Test-Path -LiteralPath $root) { throw "Output already exists: $root" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
if (-not (Test-Path -LiteralPath $PlotPython -PathType Leaf)) {
    throw "Plot Python missing: $PlotPython"
}
if ($Device -notin @('cuda', 'cpu')) { throw 'Device must be cuda or cpu' }
& $Python -c 'import torch; assert torch.cuda.is_available()' 2>$null
if ($Device -eq 'cuda' -and $LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }
& $PlotPython -c 'import matplotlib, numpy'
if ($LASTEXITCODE -ne 0) { throw 'Plotting preflight failed' }

New-Item -ItemType Directory -Path $root | Out-Null
$log = Join-Path $root 'parallel_queue.log'
$common = @('--device', $Device)
if ($Smoke) {
    $common += @('--episodes', '10', '--warmup', '5', '--block-size', '5',
                 '--rollout-steps', '8', '--validation-interval', '5',
                 '--validation-per-scenario', '1', '--id-episodes', '1',
                 '--ood-episodes', '1', '--surrogate-episodes', '1',
                 '--surrogate-epochs', '1', '--pretrain-episodes', '5',
                 '--pgd-steps', '1', '--pgd-restarts', '1')
}

function Start-Worker([int]$Seed, [string]$Phase, [string]$Method) {
    $label = if ($Method) { "seed${Seed}_${Method}" } else { "seed${Seed}_${Phase}" }
    $output = Join-Path $root "main_seed$Seed"
    $stdout = Join-Path $root "$label.stdout.log"
    $stderr = Join-Path $root "$label.stderr.log"
    $arguments = @('-u', $worker, '--output-dir', $output,
                   '--phase', $Phase, '--seed', "$Seed") + $common
    if ($Method) { $arguments += @('--method', $Method) }
    $process = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    "START label=$label pid=$($process.Id) $(Get-Date -Format o)" |
        Add-Content -LiteralPath $log
    $marker = switch ($Phase) {
        'prepare' { Join-Path $output 'PREPARE_COMPLETE.txt' }
        'branch' { Join-Path (Join-Path $output $Method) 'BRANCH_COMPLETE.txt' }
        'test' { Join-Path $output 'TEST_COMPLETE.txt' }
    }
    return [pscustomobject]@{ Label=$label; Process=$process; Marker=$marker;
                              Stderr=$stderr }
}

function Wait-Stage([object[]]$Jobs) {
    $pending = @($Jobs)
    try {
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
    } catch {
        foreach ($job in $pending) {
            $job.Process.Refresh()
            if (-not $job.Process.HasExited) {
                Stop-Process -Id $job.Process.Id -ErrorAction SilentlyContinue
                "STOPPED label=$($job.Label) due_to_stage_failure" |
                    Add-Content -LiteralPath $log
            }
        }
        throw
    }
}

try {
    $preparations = @(foreach ($seed in $seeds) { Start-Worker $seed 'prepare' '' })
    Wait-Stage $preparations
    foreach ($method in @('ppo', 'pgd_ppo')) {
        $branches = @(foreach ($seed in $seeds) {
            Start-Worker $seed 'branch' $method
        })
        Wait-Stage $branches
    }
    $tests = @(foreach ($seed in $seeds) { Start-Worker $seed 'test' '' })
    Wait-Stage $tests
    $plotStdout = Join-Path $root 'plot.stdout.log'
    $plotStderr = Join-Path $root 'plot.stderr.log'
    $plotProcess = Start-Process -FilePath $PlotPython `
        -ArgumentList @('-u', $plotter, '--run-dir', $root) `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $plotStdout -RedirectStandardError $plotStderr
    $figureMarker = Join-Path (Join-Path $root 'figures') 'FOUR_METRIC_PLOT_COMPLETE.txt'
    Wait-Stage @([pscustomobject]@{ Label='plot'; Process=$plotProcess;
        Marker=$figureMarker; Stderr=$plotStderr })
    $intervalStdout = Join-Path $root 'interval_plot.stdout.log'
    $intervalStderr = Join-Path $root 'interval_plot.stderr.log'
    $intervalProcess = Start-Process -FilePath $PlotPython `
        -ArgumentList @('-u', $intervalPlotter, '--run-dir', $root) `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $intervalStdout -RedirectStandardError $intervalStderr
    $intervalMarker = Join-Path (Join-Path $root 'figures') 'SIX_SEED_INTERVAL_PLOTS_COMPLETE.txt'
    Wait-Stage @([pscustomobject]@{ Label='interval_plot'; Process=$intervalProcess;
        Marker=$intervalMarker; Stderr=$intervalStderr })
    'QUEUE_COMPLETE' | Add-Content -LiteralPath $log
} catch {
    "QUEUE_STOPPED $($_.Exception.Message)" | Add-Content -LiteralPath $log
    throw
}
