param([string]$Source, [string]$Output)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
$deck = $null
$issues = [System.Collections.Generic.List[object]]::new()
function Inspect-Shape($shape, $slideNumber) {
    if ($shape.Type -eq 6) {
        foreach ($child in $shape.GroupItems) { Inspect-Shape $child $slideNumber }
    }
    if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
        $tr = $shape.TextFrame2.TextRange
        $height = $shape.Height - $shape.TextFrame2.MarginTop - $shape.TextFrame2.MarginBottom
        $width = $shape.Width - $shape.TextFrame2.MarginLeft - $shape.TextFrame2.MarginRight
        if (($tr.BoundHeight -gt ($height + 3)) -or ($tr.BoundWidth -gt ($width + 3))) {
            $issues.Add([pscustomobject]@{
                slide = $slideNumber; name = $shape.Name; text = $tr.Text
                height = $height; boundHeight = $tr.BoundHeight
                width = $width; boundWidth = $tr.BoundWidth
            })
        }
    }
}
try {
    $deck = $app.Presentations.Open($Source, $true, $false, $false)
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) { Inspect-Shape $shape $slide.SlideIndex }
    }
    $issues | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding UTF8
    Write-Output ('Possible text overflows: ' + $issues.Count)
    $issues | Format-Table slide, name, height, boundHeight, width, boundWidth -AutoSize
} finally {
    if ($null -ne $deck) { $deck.Close() }
    $app.Quit()
}
