@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "FEATURE_ARGS="
if "%~1"=="--openblas-static" (
    set "FEATURE_ARGS=--features openblas-static"
) else if not "%~1"=="" (
    echo Usage: %~nx0 [--openblas-static] 1>&2
    exit /b 2
)

set "FOUND=0"
for %%F in (examples\lapack_*.rs) do (
    if exist "%%F" (
        set "FOUND=1"
        echo ==^> cargo run --example %%~nF %FEATURE_ARGS%
        cargo run --quiet --example "%%~nF" %FEATURE_ARGS%
        if errorlevel 1 exit /b !errorlevel!
    )
)

if "%FOUND%"=="0" (
    echo No LAPACK examples found in examples\lapack_*.rs 1>&2
    exit /b 1
)

exit /b 0
