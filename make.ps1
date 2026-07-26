<#
.SYNOPSIS
    Odpowiednik Makefile'a dla Windowsa — GNU Make nie jest tam wbudowany.

.DESCRIPTION
    Te same cele co w Makefile, bez instalowania czegokolwiek:

        .\make.ps1 up
        .\make.ps1 logs ml-pipeline
        .\make.ps1 bootstrap-universe --train --report-out reports/first-training.json

    Skrypt sam odnajduje katalog repozytorium (dziala z dowolnego miejsca)
    i sam wykrywa interpreter Pythona: `python`, launcher `py -3` albo `python3`.

    Jesli Windows zablokuje uruchomienie (ExecutionPolicy):
        powershell -ExecutionPolicy Bypass -File .\make.ps1 up

.EXAMPLE
    .\make.ps1 help
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    # Wszystko po nazwie celu leci dalej (flagi bootstrapu, nazwa serwisu itp.).
    # Nie nazywac tego $Args — to zmienna automatyczna PowerShella.
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
$Repo = $PSScriptRoot
$ComposeFile = Join-Path $Repo 'infrastructure/docker-compose.yml'
$EnvFile = Join-Path $Repo '.env'

$script:PythonExe = $null
$script:PythonPrefix = @()

function Invoke-Compose([string[]]$ComposeArgs) {
    if (-not (Test-Path $EnvFile)) {
        throw "Brak pliku .env w $Repo — utworz go: Copy-Item .env.example .env, potem uzupelnij hasla."
    }
    & docker compose -f $ComposeFile --env-file $EnvFile @ComposeArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Initialize-Python {
    if ($script:PythonExe) { return }
    # Kolejnosc jak na Windowsie: `python`, launcher `py -3`, na koncu python3.
    foreach ($spec in @('python', 'py -3', 'python3')) {
        $parts = $spec.Split(' ')
        $exe = $parts[0]
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $prefix = @()
        if ($parts.Length -gt 1) { $prefix = @($parts[1]) }
        try {
            # `python` w Windowsie bywa aliasem Microsoft Store, ktory tylko otwiera sklep
            $probe = & $exe @prefix '-c' 'print(42)' 2>$null
        }
        catch { continue }
        if ($probe -eq '42') {
            $script:PythonExe = $exe
            $script:PythonPrefix = $prefix
            return
        }
    }
    throw 'Nie znaleziono dzialajacego Pythona 3. Zainstaluj go z python.org (zaznacz "Add to PATH").'
}

function Invoke-Python([string[]]$PyArgs) {
    Initialize-Python
    $prefix = $script:PythonPrefix
    & $script:PythonExe @prefix @PyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Show-Help {
    Write-Host ''
    Write-Host '  Cele (odpowiednik Makefile):' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '    up                   Uruchom wszystkie serwisy (docker compose up -d)'
    Write-Host '    down                 Zatrzymaj wszystkie serwisy'
    Write-Host '    build [serwis]       Zbuduj obrazy (bez argumentu: wszystkie)'
    Write-Host '    ps                   Status kontenerow'
    Write-Host '    logs [serwis]        Sledz logi (bez argumentu: wszystkie)'
    Write-Host '    test [serwis]        Testy (bez argumentu: wszystkie komponenty)'
    Write-Host '    bootstrap-universe   Backfill danych + opcjonalny trening'
    Write-Host '    verify-jetstream     Sprawdz NATS JetStream end-to-end'
    Write-Host '    helm-template        Renderuj chart Helma (dry-run)'
    Write-Host '    helm-install         Deploy przez Helma'
    Write-Host '    help                 Ta pomoc'
    Write-Host ''
    Write-Host '  Przyklady:' -ForegroundColor Cyan
    Write-Host '    .\make.ps1 up'
    Write-Host '    .\make.ps1 build market-data'
    Write-Host '    .\make.ps1 bootstrap-universe --train --report-out reports/first-training.json'
    Write-Host ''
}

function Invoke-AllTests {
    Initialize-Python
    $prefix = $script:PythonPrefix
    $dirs = @(Join-Path $Repo 'shared/trading-common')
    $dirs += (Get-ChildItem (Join-Path $Repo 'services') -Directory | ForEach-Object { $_.FullName })

    $failed = @()
    foreach ($dir in $dirs) {
        if (-not (Test-Path (Join-Path $dir 'tests'))) { continue }
        $name = Split-Path $dir -Leaf
        Write-Host "`n== $name ==" -ForegroundColor Cyan
        Push-Location $dir
        try {
            & $script:PythonExe @prefix '-m' 'pytest' 'tests/' '-q'
            if ($LASTEXITCODE -ne 0) { $failed += $name }
        }
        finally { Pop-Location }
    }
    if ($failed.Count -gt 0) {
        Write-Host "`nNIEPOWODZENIE: $($failed -join ', ')" -ForegroundColor Red
        exit 1
    }
    Write-Host "`nWszystkie testy zielone." -ForegroundColor Green
}

function Invoke-ServiceTests([string]$Service) {
    $dir = if ($Service -eq 'shared') {
        Join-Path $Repo 'shared/trading-common'
    }
    else {
        Join-Path $Repo "services/$Service"
    }
    if (-not (Test-Path $dir)) { throw "Nie ma katalogu $dir" }
    Push-Location $dir
    try { Invoke-Python @('-m', 'pytest', 'tests/', '-v') } finally { Pop-Location }
}

switch ($Target) {
    'up' { Invoke-Compose @('up', '-d') }
    'down' { Invoke-Compose @('down') }
    'ps' { Invoke-Compose @('ps') }
    'build' { Invoke-Compose (@('build') + $Rest) }
    'logs' { Invoke-Compose (@('logs', '-f', '--tail=100') + $Rest) }

    'test' {
        if ($Rest.Count -gt 0) { Invoke-ServiceTests $Rest[0] } else { Invoke-AllTests }
    }

    'bootstrap-universe' {
        Invoke-Python (@((Join-Path $Repo 'scripts/bootstrap-universe.py')) + $Rest)
    }
    'verify-jetstream' {
        Invoke-Python (@((Join-Path $Repo 'scripts/verify-jetstream.py')) + $Rest)
    }

    'helm-template' {
        & helm template trading-system (Join-Path $Repo 'infrastructure/helm') `
            -f (Join-Path $Repo 'infrastructure/helm/values.yaml')
    }
    'helm-install' {
        & helm upgrade --install trading-system (Join-Path $Repo 'infrastructure/helm') `
            -f (Join-Path $Repo 'infrastructure/helm/values.yaml')
    }

    default {
        if ($Target -ne 'help') { Write-Host "Nieznany cel: $Target" -ForegroundColor Yellow }
        Show-Help
    }
}
