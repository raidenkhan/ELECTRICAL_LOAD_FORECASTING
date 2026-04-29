$files = Get-ChildItem -Path src -Filter *.tsx -Recurse
$allDeps = New-Object System.Collections.Generic.HashSet[string]
foreach ($file in $files) {
    $lines = Get-Content $file.FullName
    foreach ($line in $lines) {
        if ($line.Contains("from '") -or $line.Contains('from "')) {
            $parts = $line.Split("'")
            if ($parts.Length -lt 2) { $parts = $line.Split('"') }
            if ($parts.Length -ge 2) {
                $dep = $parts[1]
                if ($dep.Length -gt 0 -and (-not $dep.StartsWith(".")) -and (-not $dep.StartsWith("@/"))) {
                    $allDeps.Add($dep) | Out-Null
                }
            }
        }
    }
}
$allDeps | Sort-Object | Set-Content dependencies_found.tmp
