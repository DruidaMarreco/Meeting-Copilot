# setup-github.ps1
# Run this once in PowerShell from inside the meeting-copilot folder.
# It cleans stale git lock files and pushes to GitHub.
#
# Usage:
#   cd "C:\Users\DELL\Documents\Projetos Pessoais\meeting-copilot"
#   .\setup-github.ps1 -RepoUrl https://github.com/YOUR_USERNAME/meeting-copilot.git

param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

Write-Host "`n=== Meeting Copilot — GitHub Setup ===" -ForegroundColor Cyan

# 1. Remove stale lock files left by Linux git
Write-Host "`n[1] Cleaning stale lock files..." -ForegroundColor Yellow
$locks = Get-ChildItem -Path ".git" -Filter "*.lock" -Recurse -ErrorAction SilentlyContinue
foreach ($lock in $locks) {
    Remove-Item $lock.FullName -Force
    Write-Host "    Removed: $($lock.Name)"
}
if (-not $locks) { Write-Host "    None found — already clean" }

# 2. Configure git identity (update these if needed)
Write-Host "`n[2] Setting git identity..." -ForegroundColor Yellow
git config user.name "Simao Miguel"
git config user.email "simao.tc.miguel@gmail.com"
Write-Host "    OK"

# 3. Rename branch master -> main (GitHub default)
Write-Host "`n[3] Renaming branch master -> main..." -ForegroundColor Yellow
git branch -m master main
Write-Host "    OK"

# 4. Add remote
Write-Host "`n[4] Adding remote origin..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin $RepoUrl
Write-Host "    $RepoUrl"

# 5. Push
Write-Host "`n[5] Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`n=== Done! Your repo is live at $RepoUrl ===" -ForegroundColor Green
Write-Host "Next: open a Zoom call and run: python -m audio" -ForegroundColor Cyan
