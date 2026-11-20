param([ValidateSet('Stage','Prune')][string]$Phase = 'Stage')
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$dataRoot = [IO.Path]::GetFullPath((Join-Path $repo 'data'))
$rawTarget = Join-Path $dataRoot 'raw/child_crime_exposure'
$processedTarget = Join-Path $dataRoot 'processed/child_crime_exposure'
$auditRoot = Join-Path $repo 'artifacts/data_cleanup_20261117'

function Assert-DataPath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($dataRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target is outside the dataset directory: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) {
        $item = Get-Item -LiteralPath $resolved -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked path: $resolved" }
    }
    return $resolved
}

if ($Phase -eq 'Stage') {
    if (Test-Path -LiteralPath $rawTarget) { throw 'Already staged; inspect existing cleanup manifest.' }
    $oldRaw = Join-Path $dataRoot 'raw/아동범죄_행정동'
    $fixed = @('서울시_5대범죄_2014_2025_원자료.csv', '101_DT_1JC1501_20261106110150.xls',
        '사업체 및 종사자 밀도.xls', '서울시 인구이동(2015년 이후) 통계.csv',
        '학생변동상황(전출·전입)_20261110103410.xlsx',
        '서울시 유흥주점영업 인허가 정보.csv', '서울시 단란주점영업 인허가 정보.csv')
    $kept = @(Get-ChildItem -LiteralPath $oldRaw -Recurse -File | Where-Object {
        $_.Name -in $fixed -or $_.Name -like '*연령별인구현황_연간.csv' -or
        $_.Name -like '출생사망인구수_*.xls' -or $_.Name -like '파일설계서*.xlsx' -or
        ($_.Name -match '^(201[4-9]|202[0-4])_연간자료_.*\.csv$')
    })
    if ($kept.Count -ne 38) { throw "Expected 38 raw inputs, got $($kept.Count)" }
    if (@($kept | Group-Object Name | Where-Object Count -gt 1).Count) { throw 'Duplicate flattened filename.' }
    New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
    $all = @(Get-ChildItem -LiteralPath $dataRoot -Recurse -File -Force)
    $keepPaths = @($kept.FullName)
    $manifest = @($all | ForEach-Object {
        [PSCustomObject]@{path=$_.FullName.Substring($repo.Length+1); bytes=$_.Length;
            action=$(if ($_.FullName -in $keepPaths) {'keep_raw'} elseif ($_.Name -eq '.gitkeep' -and $_.DirectoryName -in @((Join-Path $dataRoot 'raw'),(Join-Path $dataRoot 'processed'),(Join-Path $dataRoot 'interim'))) {'keep_placeholder'} else {'delete'})}
    })
    $manifest | Export-Csv -LiteralPath (Join-Path $auditRoot 'original_data_manifest.csv') -NoTypeInformation -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $repo 'artifacts/child_crime_borough_v2/panel.csv') -Destination (Join-Path $auditRoot 'before_panel.csv')
    Copy-Item -LiteralPath (Join-Path $repo 'artifacts/child_population_priority_v1/priority_features.csv') -Destination (Join-Path $auditRoot 'before_priority_features.csv')
    New-Item -ItemType Directory -Path $rawTarget -Force | Out-Null
    $hashes = @($kept | ForEach-Object {
        $source = Assert-DataPath $_.FullName
        $target = Assert-DataPath (Join-Path $rawTarget $_.Name)
        $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
        Move-Item -LiteralPath $source -Destination $target
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash) { throw "Hash mismatch: $target" }
        [PSCustomObject]@{original=$_.FullName; target=$target; sha256=$hash; bytes=$_.Length}
    })
    $hashes | Export-Csv -LiteralPath (Join-Path $auditRoot 'retained_raw_hashes.csv') -Encoding UTF8 -NoTypeInformation
    New-Item -ItemType Directory -Path $processedTarget -Force | Out-Null
    foreach ($entry in @(@('child_crime_borough_v2','borough_panel'),@('child_population_priority_v1','population_features'))) {
        $source = [IO.Path]::GetFullPath((Join-Path $repo ('artifacts/' + $entry[0])))
        $expected = [IO.Path]::GetFullPath((Join-Path $repo ('artifacts/' + $entry[0])))
        if ($source -ne $expected -or -not $source.StartsWith((Join-Path $repo 'artifacts') + [IO.Path]::DirectorySeparatorChar)) { throw 'Invalid artifact source' }
        $target = Assert-DataPath (Join-Path $processedTarget $entry[1])
        Move-Item -LiteralPath $source -Destination $target
    }
    Write-Output "Staged $($kept.Count) verified raw files; no unrelated datasets deleted yet."
} else {
    $verification = Join-Path $auditRoot 'verification.json'
    if (-not (Test-Path -LiteralPath $verification)) { throw 'Run the data and forecast verification before pruning.' }
    $check = Get-Content -LiteralPath $verification -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $check.passed) { throw 'Data verification did not pass.' }
    $manifest = @(Import-Csv -LiteralPath (Join-Path $auditRoot 'original_data_manifest.csv') -Encoding UTF8)
    $deleted = 0; $bytes = [long]0
    foreach ($entry in $manifest | Where-Object action -eq 'delete') {
        $path = Assert-DataPath (Join-Path $repo $entry.path)
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $item = Get-Item -LiteralPath $path -Force
        if ($item.PSIsContainer -or $item.Length -ne [long]$entry.bytes) { throw "Changed deletion target: $path" }
        # Exact inventory targets only; never recursively delete a dataset root.
        Remove-Item -LiteralPath $path -Force
        $deleted++; $bytes += [long]$entry.bytes
    }
    foreach ($folder in Get-ChildItem -LiteralPath $dataRoot -Directory -Recurse -Force | Sort-Object { $_.FullName.Length } -Descending) {
        if ($folder.FullName -in @((Join-Path $dataRoot 'raw'),(Join-Path $dataRoot 'processed'),(Join-Path $dataRoot 'interim'))) { continue }
        $path = Assert-DataPath $folder.FullName
        if (@(Get-ChildItem -LiteralPath $path -Force).Count -eq 0) { Remove-Item -LiteralPath $path -Force }
    }
    [PSCustomObject]@{deleted_files=$deleted; deleted_bytes=$bytes; retained_raw_files=38; permanent_deletion=$true; finished_at=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $auditRoot 'cleanup_result.json') -Encoding UTF8
    Write-Output "Deleted $deleted files ($([math]::Round($bytes/1GB,3)) GiB). Retained data verified before removal."
}
