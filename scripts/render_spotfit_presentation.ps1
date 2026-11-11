param([string]$Source, [string]$Output)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open($Source, $true, $false, $false)
    $deck.Export($Output, 'PNG', 1600, 900)
    Write-Output ('Rendered slides: ' + $deck.Slides.Count)
} finally {
    if ($null -ne $deck) { $deck.Close() }
    $app.Quit()
}
