@echo off
REM Local shim to force wandb sweep sub-runs to use the project virtualenv Python.
REM Generated automatically. Delete when no longer needed.
"%~dp0.venv\Scripts\python.exe" %*
