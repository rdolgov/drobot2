param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$packageDirectory = $PSScriptRoot
$archivePath = Join-Path $packageDirectory 'ik-diagonal-bigstride-v2-pi-baseline.zip'
$expectedHash = '732509dbf79e78c8d97faf8806859e0baf8931c4174032291b0e070d9a13b2d5'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'Package checksum changed; regenerate the reviewed bundle.'
}
& scp $archivePath 'rd@pi5-dog.local:/home/rd/ik-diagonal-bigstride-v2-pi-baseline.zip'
if ($LASTEXITCODE -ne 0) { throw 'Archive upload failed.' }
& scp (Join-Path $packageDirectory 'apply-on-pi.py') 'rd@pi5-dog.local:/home/rd/apply-ik-diagonal-v2.py'
if ($LASTEXITCODE -ne 0) { throw 'Installer upload failed.' }
$remoteCommand = 'python3 /home/rd/apply-ik-diagonal-v2.py --repo /home/rd/drobot2 --archive /home/rd/ik-diagonal-bigstride-v2-pi-baseline.zip --sha256 732509dbf79e78c8d97faf8806859e0baf8931c4174032291b0e070d9a13b2d5'
if ($Apply) { $remoteCommand += ' --apply' }
& ssh 'rd@pi5-dog.local' $remoteCommand
if ($LASTEXITCODE -ne 0) { throw 'Pi package operation failed; inspect its message before retrying.' }
