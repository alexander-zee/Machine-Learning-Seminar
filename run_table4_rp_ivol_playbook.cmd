@echo off
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo Running RP IdioVol playbook...
python "%~dp0run_table4_rp_ivol_playbook.py" %*
echo.
echo Exit code: %ERRORLEVEL%
pause
exit /b %ERRORLEVEL%
