param(
    [string]$Python = 'D:\Anaconda_envs\envs\migrationproject\python.exe',
    [string]$RunName = 'ood_pgd_20261012',
    [int]$PilotPid = 43484
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$entry = Join-Path $PSScriptRoot 'run_ood_pgd_parallel_worker.py'
$root = Join-Path $repo (Join-Path 'runs' $RunName)
$log = Join-Path $root 'six_parallel_queue.log'
$pilotMarker = Join-Path $root 'pilot_seed1\TRAINING_COMPLETE.txt'

function Start-Worker([int]$Seed, [string]$Phase, [string]$Method) {
    $label = if ($Method) { "seed${Seed}_${Method}" } else { "seed${Seed}_${Phase}" }
    $output = Join-Path $root "main_seed$Seed"
    if ($Phase -eq 'prepare' -and (Test-Path -LiteralPath $output)) {
        throw "Output already exists: $output"
    }
    $stdout = Join-Path $root "$label.stdout.log"
    $stderr = Join-Path $root "$label.stderr.log"
    $arguments = @('-u', $entry, '--output-dir', $output, '--phase', $Phase,
                   '--seed', "$Seed", '--episodes', '20000',
                   '--warmup', '2000', '--device', 'cuda')
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
                if (-not (Test-Path -LiteralPath $job.Marker)) {
                    "FAILED label=$($job.Label) launcher_exit=$($job.Process.ExitCode) missing_final_marker" |
                        Add-Content -LiteralPath $log
                    throw "Worker failed: $($job.Label). See $($job.Stderr)"
                }
                if ($job.Process.ExitCode -ne 0) {
                    "WARNING label=$($job.Label) launcher_exit=$($job.Process.ExitCode) final_marker_present" |
                        Add-Content -LiteralPath $log
                }
                "COMPLETE label=$($job.Label) $(Get-Date -Format o)" |
                    Add-Content -LiteralPath $log
                $pending = @($pending | Where-Object { $_.Label -ne $job.Label })
            }
            if ($pending.Count -gt 0) { Start-Sleep -Seconds 30 }
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
    "WAIT_PILOT pid=$PilotPid $(Get-Date -Format o)" | Add-Content -LiteralPath $log
    while (-not (Test-Path -LiteralPath $pilotMarker)) {
        if (-not (Get-Process -Id $PilotPid -ErrorAction SilentlyContinue)) {
            throw 'Pilot ended without TRAINING_COMPLETE.txt'
        }
        Start-Sleep -Seconds 30
    }
    'PILOT_COMPLETE' | Add-Content -LiteralPath $log
    $preparations = @(foreach ($seed in 1..3) { Start-Worker $seed 'prepare' '' })
    Wait-Stage $preparations
    'SIX_BRANCHES_STARTING' | Add-Content -LiteralPath $log
    $branches = @(foreach ($seed in 1..3) {
        foreach ($method in @('ppo', 'pgd_ppo')) {
            Start-Worker $seed 'branch' $method
        }
    })
    Wait-Stage $branches
    $tests = @(foreach ($seed in 1..3) { Start-Worker $seed 'test' '' })
    Wait-Stage $tests
    'QUEUE_COMPLETE' | Add-Content -LiteralPath $log
} catch {
    "QUEUE_STOPPED $($_.Exception.Message)" | Add-Content -LiteralPath $log
    throw
}
