param([Parameter(Mandatory=$true)][string]$Source, [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$app=New-Object -ComObject PowerPoint.Application
$priorAlerts=$app.DisplayAlerts
$app.DisplayAlerts=1
$deck=$null
try {
    $deck=$app.Presentations.Open($Source,$true,$false,$false)
    $deck.Export($Output,'PNG',1600,900)
    $pdf=[System.IO.Path]::ChangeExtension($Source,'.pdf')
    $deck.SaveAs($pdf,32)
    $issues=New-Object System.Collections.Generic.List[object]
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame2.HasText) {
                $height=$shape.TextFrame2.TextRange.BoundHeight
                $available=$shape.Height-$shape.TextFrame2.MarginTop-$shape.TextFrame2.MarginBottom
                if ($height -gt ($available+4)) {
                    $issues.Add([pscustomobject]@{slide=$slide.SlideIndex;shape=$shape.Name;excess=[Math]::Round($height-$available,1);text=$shape.TextFrame2.TextRange.Text})
                }
            }
            if ($shape.HasTable) {
                foreach ($row in $shape.Table.Rows) {
                    foreach ($cell in $row.Cells) {
                        $tf=$cell.Shape.TextFrame2
                        if ($tf.HasText) {
                            $height=$tf.TextRange.BoundHeight
                            $available=$cell.Shape.Height-$tf.MarginTop-$tf.MarginBottom
                            if ($height -gt ($available+4)) {
                                $issues.Add([pscustomobject]@{slide=$slide.SlideIndex;shape='table_cell';excess=[Math]::Round($height-$available,1);text=$tf.TextRange.Text})
                            }
                        }
                    }
                }
            }
        }
    }
    ConvertTo-Json -InputObject @($issues.ToArray()) -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $Output 'overflow_check.json')
    Write-Output ('Rendered '+$deck.Slides.Count+' slides; overflow candidates: '+$issues.Count)
    $issues | Format-Table slide,shape,excess,text -Wrap
    Write-Output $pdf
} finally {
    if ($null -ne $deck) {$deck.Close()}
    $app.DisplayAlerts=$priorAlerts
    # Do not close any presentation the user already had open.
    if ($app.Presentations.Count -eq 0) {$app.Quit()}
}
