$p = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | Where-Object { $_.CommandLine -match 'monikavoice' }
$proc = Get-Process -Id $p.ProcessId
"PID: " + $proc.Id
"WorkingSet MB: " + [math]::Round($proc.WorkingSet64/1MB)
"Commit MB: " + [math]::Round($proc.PagedMemorySize64/1MB)
"TotalCPU s: " + [math]::Round($proc.TotalProcessorTime.TotalSeconds,1)
"Uptime min: " + [math]::Round(((Get-Date) - $proc.StartTime).TotalMinutes,1)
$c1 = $proc.TotalProcessorTime.TotalSeconds
Start-Sleep -Seconds 10
$c2 = (Get-Process -Id $proc.Id).TotalProcessorTime.TotalSeconds
"IdleCPU pct: " + [math]::Round(($c2-$c1)/10/[Environment]::ProcessorCount*100,2)
"Cores: " + [Environment]::ProcessorCount
