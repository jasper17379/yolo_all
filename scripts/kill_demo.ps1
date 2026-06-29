@echo off
REM 强制结束可能残留的 demo / 训练 Python 进程
echo 正在查找 vision demo 相关 Python 进程...
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'demo\.py|src\.api\.server|src\.train\.trainer|verify_pipeline' } | ForEach-Object { Write-Host ('结束 PID ' + $_.ProcessId + ': ' + $_.CommandLine); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
echo 完成。若 OpenCV 窗口仍可见，请手动关闭或注销桌面会话。
