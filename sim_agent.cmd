@echo off
rem opencode in this repo, driving the simulated arm (artist-arm-sim) instead of the real one. Double-click to start.
cd /d "%~dp0"
set "OPENCODE_CONFIG_CONTENT={"mcp":{"artist-arm":{"enabled":false},"artist-arm-sim":{"enabled":true}}}"
if "%~1"=="--print-config" (echo %OPENCODE_CONFIG_CONTENT%& exit /b 0)
call "%APPDATA%\npm\opencode.cmd"
