$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
Write-Output ("TotalRAM_GB: {0}" -f [math]::Round($cs.TotalPhysicalMemory/1GB, 1))
Write-Output ("FreeRAM_GB: {0}"  -f [math]::Round($os.FreePhysicalMemory/1MB, 1))
Write-Output ("TotalVirt_GB: {0}" -f [math]::Round($os.TotalVirtualMemorySize/1MB, 1))
Write-Output ("FreeVirt_GB: {0}"  -f [math]::Round($os.FreeVirtualMemory/1MB, 1))
Write-Output ""
Write-Output "Top-10 memory processes:"
Get-Process | Sort-Object -Descending WorkingSet | Select-Object -First 10 ProcessName, @{N="MemMB";E={[math]::Round($_.WorkingSet/1MB,0)}} | Format-Table -AutoSize
