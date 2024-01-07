@echo off

:: Define variables
set "project_path=%CD%\"
set "env_path=%project_path%build_ARIMA_model"
echo %env_path%
echo "-----------------"

start cmd /k "conda activate %env_path%"

goto :eof