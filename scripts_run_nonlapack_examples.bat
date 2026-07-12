@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "FOUND=0"

for %%F in (examples\*.rs) do (
    if exist "%%F" (
        set "EXAMPLE=%%~nF"
        if /I not "!EXAMPLE:~0,7!"=="lapack_" (
            set "FOUND=1"
            echo ==> cargo run --example !EXAMPLE!
            cargo run --quiet --example "!EXAMPLE!"
            if errorlevel 1 exit /b !errorlevel!
        )
    )
)

if "%FOUND%"=="0" (
    echo No non-LAPACK examples found in examples\*.rs 1>&2
    exit /b 1
)

exit /b 0