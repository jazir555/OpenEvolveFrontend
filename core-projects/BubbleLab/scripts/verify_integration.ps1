# verify_integration.ps1
# Unified verification harness for the OpenEvolve <-> BubbleLab integration.
# Runs every known verification suite, captures exit codes, prints a summary
# table, and exits non-zero if any suite FAILs. Suites whose tooling is missing
# are reported as SKIPPED rather than failing the whole run.

[CmdletBinding()]
param(
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'

# Resolve the BubbleLab root as the parent of this script's directory.
$BubbleLabRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
# Repo root is two levels up (OpenEvolveFrontend).
$RepoRoot = (Resolve-Path (Join-Path $BubbleLabRoot '..' '..')).Path

function Test-CommandExists {
    param([string]$Name)
    try {
        $null = Get-Command $Name -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Invoke-Suite {
    param(
        [string]$Suite,
        [string]$Dir,
        [string]$Command,
        [string]$Expected
    )

    $absDir = Join-Path $BubbleLabRoot $Dir
    if (-not (Test-Path $absDir)) {
        return [PSCustomObject]@{
            Suite    = $Suite
            Command  = $Command
            Result   = 'SKIP'
            Note     = "Directory not found: $Dir"
            Elapsed  = 0
        }
    }

    # Pick the executable to check for existence (first token of the command).
    $exe = ($Command -split '\s+')[0]
    if ($exe -match '^[.\/\\]') { $exe = 'cmd' }  # local script path; assume present
    if (-not (Test-CommandExists $exe)) {
        return [PSCustomObject]@{
            Suite    = $Suite
            Command  = $Command
            Result   = 'SKIP'
            Note     = "Tool not found: $exe"
            Elapsed  = 0
        }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = 'cmd.exe'
    $psi.Arguments = "/c cd /d `"$absDir`" && $Command"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
    } catch {
        $sw.Stop()
        return [PSCustomObject]@{
            Suite    = $Suite
            Command  = $Command
            Result   = 'SKIP'
            Note     = "Could not start process: $_"
            Elapsed  = 0
        }
    }

    $output = New-Object System.Text.StringBuilder
    $outTask = $proc.StandardOutput.ReadToEndAsync()
    $errTask = $proc.StandardError.ReadToEndAsync()

    $exited = $proc.WaitForExit($TimeoutSeconds * 1000)
    if (-not $exited) {
        try { $proc.Kill() } catch {}
        $sw.Stop()
        return [PSCustomObject]@{
            Suite    = $Suite
            Command  = $Command
            Result   = 'FAIL'
            Note     = "Timed out after ${TimeoutSeconds}s"
            Elapsed  = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        }
    }
    $out = $outTask.Result
    $err = $errTask.Result
    $sw.Stop()
    $all = "$out`n$err" -split "`n"
    # Prefer the pytest summary line (e.g. "1 failed, 76 passed, 28 errors in 11s").
    $summaryLine = $all | Where-Object { $_ -match '\b(passed|failed|error|skipped)\b.*\bin \d' } | Select-Object -Last 1
    if (-not $summaryLine) {
        $summaryLine = ($all | Where-Object { $_.Trim() -ne '' } | Select-Object -Last 3) -join ' / '
    }

    $result = if ($proc.ExitCode -eq 0) { 'PASS' } else { 'FAIL' }
    $note = if ($result -eq 'PASS') {
        if ($Expected) { "expected: $Expected" } else { "exit 0" }
    } else {
        "exit $($proc.ExitCode) | $($summaryLine.Trim())"
    }

    return [PSCustomObject]@{
        Suite    = $Suite
        Command  = $Command
        Result   = $result
        Note     = $note
        Elapsed  = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    }
}

$suites = @(
    [PSCustomObject]@{
        Suite    = 'OpenEvolve lib (pytest)'
        Dir      = '../openevolve'
        Command  = 'python -m pytest tests/test_mock_evolution.py -q -p no:pytest_ethereum'
        Expected = '10 passed'
    },
    [PSCustomObject]@{
        Suite    = 'OpenEvolve API (pytest)'
        Dir      = 'services/openevolve-api'
        Command  = 'python -m pytest tests/ -q -p no:pytest_ethereum'
        Expected = 'green'
    },
    [PSCustomObject]@{
        Suite    = 'Service boot smoke (launch_demo)'
        Dir      = 'services/openevolve-api'
        Command  = 'python scripts/launch_demo.py'
        Expected = 'PASS'
    },
    [PSCustomObject]@{
        Suite    = 'Service boot smoke (proxy_path_test)'
        Dir      = 'services/openevolve-api'
        Command  = 'python scripts/proxy_path_test.py'
        Expected = 'PASS'
    },
    [PSCustomObject]@{
        Suite    = 'TS integration (tsc)'
        Dir      = 'integrations/openevolve'
        Command  = 'npx tsc --noEmit'
        Expected = '0 errors'
    },
    [PSCustomObject]@{
        Suite    = 'TS integration (test:e2e)'
        Dir      = 'integrations/openevolve'
        Command  = 'npm run test:e2e'
        Expected = '8/8'
    },
    [PSCustomObject]@{
        Suite    = 'TS integration (test:bubbles)'
        Dir      = 'integrations/openevolve'
        Command  = 'npm run test:bubbles'
        Expected = '9/9'
    },
    [PSCustomObject]@{
        Suite    = 'TS bubble-core (tsc)'
        Dir      = 'packages/bubble-core'
        Command  = 'npx tsc --noEmit'
        Expected = '0 errors'
    }
)

Write-Host "`n=== OpenEvolve <-> BubbleLab Integration Verification ===" -ForegroundColor Cyan
Write-Host "BubbleLab root: $BubbleLabRoot`n"

$results = @()
foreach ($s in $suites) {
    Write-Host "Running: $($s.Suite) ..." -NoNewline
    $r = Invoke-Suite -Suite $s.Suite -Dir $s.Dir -Command $s.Command -Expected $s.Expected
    $results += $r
    $color = switch ($r.Result) {
        'PASS'  { 'Green' }
        'SKIP'  { 'Yellow' }
        'FAIL'  { 'Red' }
        default { 'White' }
    }
    Write-Host " $($r.Result)" -ForegroundColor $color
}

# Print table
Write-Host "`nSUITE                                    | COMMAND                                  | RESULT | NOTE"
Write-Host ('-' * 110)
foreach ($r in $results) {
    $suite = $r.Suite.PadRight(40).Substring(0, 40)
    $cmd = $r.Command.PadRight(40).Substring(0, 40)
    $res = $r.Result.PadRight(6)
    Write-Host "$suite | $cmd | $res | $($r.Note)"
}

$total = $results.Count
$passed = ($results | Where-Object { $_.Result -eq 'PASS' }).Count
$skipped = ($results | Where-Object { $_.Result -eq 'SKIP' }).Count
$failed = ($results | Where-Object { $_.Result -eq 'FAIL' }).Count

Write-Host "`nSummary: $passed passed, $failed failed, $skipped skipped (of $total suites)"
if ($failed -gt 0) {
    Write-Host "`nINTEGRATION STATUS: RED" -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nINTEGRATION STATUS: GREEN" -ForegroundColor Green
    exit 0
}
