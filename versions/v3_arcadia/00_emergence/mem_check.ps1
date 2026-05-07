Get-Process -Name 'ollama','ollama_llama_server','ollama app' -ErrorAction SilentlyContinue | Select-Object Name, Id, @{N='MemMB'; E={[math]::Round($_.WorkingSet/1MB, 0)}} | Format-Table -AutoSize
