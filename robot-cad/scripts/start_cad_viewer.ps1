[CmdletBinding()]
param(
    [string]$File = "exports/step/upper_arm_st3215_fit_preview.step",
    [switch]$Markup
)

$ErrorActionPreference = "Stop"

function Get-ViewerServerInfo {
    param([int]$Port)

    try {
        return Invoke-RestMethod `
            -Uri "http://127.0.0.1:$Port/__cad/server" `
            -TimeoutSec 1
    } catch {
        return $null
    }
}

function Test-LocalTcpPort {
    param([int]$Port)

    $client = [Net.Sockets.TcpClient]::new()
    try {
        $connection = $client.ConnectAsync("127.0.0.1", $Port)
        if (-not $connection.Wait(200)) {
            return $false
        }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Activate-ViewerDirectory {
    param(
        [int]$Port,
        [string]$Directory
    )

    $encodedDirectory = [uri]::EscapeDataString($Directory)
    $activationUri = (
        "http://127.0.0.1:$Port/__cad/directory/activate" +
        "?dir=$encodedDirectory"
    )
    $result = Invoke-RestMethod `
        -Method Post `
        -Uri $activationUri `
        -TimeoutSec 5
    if (-not $result.ok) {
        throw "CAD Viewer did not activate $Directory."
    }
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$requestedPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $File))
$rootPrefix = $projectRoot.TrimEnd("\") + "\"
if (-not $requestedPath.StartsWith(
    $rootPrefix,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "The requested CAD file must be inside $projectRoot."
}
if (-not (Test-Path -LiteralPath $requestedPath)) {
    throw "CAD file not found: $requestedPath. Run scripts\generate_cad.ps1 first."
}

$viewerRoot = Join-Path $projectRoot "tools\cad-viewer-markup\runtime"
$serverPath = Join-Path $viewerRoot "backend\server.mjs"
$viewerPackagePath = Join-Path $viewerRoot "package.json"
$viewerIndexPath = Join-Path $viewerRoot "dist\index.html"
foreach ($requiredPath in @($serverPath, $viewerPackagePath, $viewerIndexPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Project-owned CAD Viewer runtime is incomplete: $requiredPath"
    }
}

$viewerPackage = Get-Content -LiteralPath $viewerPackagePath -Raw |
    ConvertFrom-Json
$viewerVersion = [string]$viewerPackage.version
if (-not $viewerVersion.Contains("drobot2-markup")) {
    throw "Project CAD Viewer runtime has an unexpected identity: $viewerVersion"
}

$runtimeRoot = Join-Path $projectRoot ".cad-viewer"
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

$viewerPort = $null
$freePort = $null
$launchAction = "reuse"
foreach ($port in 4178..4241) {
    if (-not (Test-LocalTcpPort -Port $port)) {
        $freePort = $port
        break
    }

    $serverInfo = Get-ViewerServerInfo -Port $port
    if (
        $serverInfo -and
        $serverInfo.app -eq "cad-viewer" -and
        $serverInfo.viewerVersion -eq $viewerVersion -and
        $serverInfo.serverMode -eq "drobot2-markup"
    ) {
        Activate-ViewerDirectory -Port $port -Directory $projectRoot
        $viewerPort = $port
        break
    }
}

if (-not $viewerPort) {
    if (-not $freePort) {
        throw "No free local CAD Viewer port was found from 4178 through 4241."
    }

    $viewerPort = $freePort
    $launchAction = "start"
    $launchId = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $stdoutPath = Join-Path $runtimeRoot "viewer-$launchId.stdout.log"
    $stderrPath = Join-Path $runtimeRoot "viewer-$launchId.stderr.log"
    $nodeCommand = (Get-Command node.exe -ErrorAction Stop).Source
    $serverArguments = (
        "`"$serverPath`" --host 127.0.0.1 --port $viewerPort " +
        "--dir `"$projectRoot`" --shutdown-after 12h"
    )

    $previousViewerMode = $env:VIEWER_AGENT_START_MODE
    $env:VIEWER_AGENT_START_MODE = "drobot2-markup"
    try {
        $viewerProcess = Start-Process `
            -FilePath $nodeCommand `
            -ArgumentList $serverArguments `
            -WorkingDirectory $viewerRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
    } finally {
        $env:VIEWER_AGENT_START_MODE = $previousViewerMode
    }

    $serverInfo = $null
    $readyDeadline = [DateTime]::UtcNow.AddSeconds(20)
    while (-not $serverInfo -and [DateTime]::UtcNow -lt $readyDeadline) {
        if ($viewerProcess.HasExited) {
            $errorText = if (Test-Path -LiteralPath $stderrPath) {
                Get-Content -LiteralPath $stderrPath -Raw
            } else {
                ""
            }
            throw "Project CAD Viewer exited before becoming ready. $errorText"
        }
        $candidateInfo = Get-ViewerServerInfo -Port $viewerPort
        if (
            $candidateInfo -and
            $candidateInfo.app -eq "cad-viewer" -and
            $candidateInfo.viewerVersion -eq $viewerVersion -and
            $candidateInfo.serverMode -eq "drobot2-markup"
        ) {
            $serverInfo = $candidateInfo
            break
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $serverInfo) {
        throw "Project CAD Viewer did not become ready on port $viewerPort."
    }
}

$rootUri = [uri]($projectRoot.TrimEnd("\") + "\")
$fileUri = [uri]$requestedPath
$relativeFile = [uri]::UnescapeDataString(
    $rootUri.MakeRelativeUri($fileUri).ToString()
).Replace("\", "/")
$encodedRoot = [uri]::EscapeDataString($projectRoot)
$encodedFile = [uri]::EscapeDataString($relativeFile)
$reviewUrl = (
    "http://127.0.0.1:$viewerPort/" +
    "?dir=$encodedRoot&file=$encodedFile"
)
if ($Markup) {
    $reviewUrl += "&markup=three-view"
}

[pscustomobject]@{
    Url = $reviewUrl
    File = $relativeFile
    Markup = [bool]$Markup
    ViewerRoot = $viewerRoot
    ViewerMode = "project-runtime"
    ViewerVersion = $viewerVersion
    MarkupTool = "orthographic"
    Action = $launchAction
}
