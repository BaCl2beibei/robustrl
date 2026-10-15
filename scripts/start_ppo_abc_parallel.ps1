param(
    [string]$Python = 'D:\Anaconda_envs\envs\migrationproject\python.exe',
    [string]$PlotPython = 'D:\ProgramData\anaconda3\python.exe',
    [string]$RunName = 'ppo_c_seed1_3000',
    [int]$Seed = 1
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$root = Join-Path $repo (Join-Path 'runs' $RunName)
$entry = Join-Path $PSScriptRoot 'run_ppo_abc.py'
$plot = Join-Path $PSScriptRoot 'plot_ppo_abc.py'
if (Test-Path -LiteralPath $root) { throw "Output already exists: $root" }
& $Python -c 'import torch; assert torch.cuda.is_available()'
if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }
& $PlotPython -c 'import matplotlib, numpy'
if ($LASTEXITCODE -ne 0) { throw 'Plotting preflight failed' }
New-Item -ItemType Directory -Path $root | Out-Null
$log = Join-Path $root 'parallel_queue.log'

function Start-Worker([string]$Label, [string]$Executable, [string[]]$Arguments,
                      [string]$Marker) {
    $stdout = Join-Path $root "$Label.stdout.log"
    $stderr = Join-Path $root "$Label.stderr.log"
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    "START label=$Label pid=$($process.Id) $(Get-Date -Format o)" |
        Add-Content -LiteralPath $log
    return [pscustomobject]@{ Label=$Label; Process=$process; Marker=$Marker;
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
    $training = Start-Worker 'c_hierarchical_aux' $Python @(
        '-u', $entry, '--output-dir', $root, '--phase', 'train',
        '--seed', "$Seed", '--episodes', '3000', '--device', 'cuda') `
        (Join-Path (Join-Path $root 'c_hierarchical_aux') 'BRANCH_COMPLETE.txt')
    Wait-Stage @($training)
    $test = Start-Worker 'test' $Python @('-u', $entry, '--output-dir', $root,
        '--phase', 'test', '--seed', "$Seed", '--device', 'cuda') `
        (Join-Path (Join-Path $root 'test') 'TEST_COMPLETE.txt')
    Wait-Stage @($test)
    $figure = Start-Worker 'plot' $PlotPython @('-u', $plot, '--run-dir', $root) `
        (Join-Path (Join-Path $root 'test') 'C_VS_BASELINES_PLOT_COMPLETE.txt')
    Wait-Stage @($figure)
    'QUEUE_COMPLETE' | Add-Content -LiteralPath $log
} catch {
    "QUEUE_STOPPED $($_.Exception.Message)" | Add-Content -LiteralPath $log
    throw
}
