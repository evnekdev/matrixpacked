@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "FEATURE_ARGS="
if "%~1"=="--openblas-static" (
    set "FEATURE_ARGS=--features openblas-static"
) else if "%~1"=="--intel-mkl-static" (
    set "FEATURE_ARGS=--features intel-mkl-static"
) else if not "%~1"=="" (
    echo Usage: %~nx0 [--openblas-static^|--intel-mkl-static] 1>&2
    exit /b 2
)

set "FOUND=0"

for %%F in (examples\*.rs) do (
    if exist "%%F" (
        set "EXAMPLE=%%~nF"
        if /I not "!EXAMPLE:~0,7!"=="lapack_" (
            set "FOUND=1"
            set "EXAMPLE_FEATURE_ARGS=!FEATURE_ARGS!"
            if /I "!EXAMPLE:~0,9!"=="nalgebra_" (
                if defined FEATURE_ARGS (
                    if "%~1"=="--openblas-static" set "EXAMPLE_FEATURE_ARGS=--features openblas-static,nalgebra-interop"
                    if "%~1"=="--intel-mkl-static" set "EXAMPLE_FEATURE_ARGS=--features intel-mkl-static,nalgebra-interop"
                ) else (
                    set "EXAMPLE_FEATURE_ARGS=--features nalgebra-interop"
                )
            )
            echo ==^> cargo run --example !EXAMPLE! !EXAMPLE_FEATURE_ARGS!
            cargo run --quiet --example "!EXAMPLE!" !EXAMPLE_FEATURE_ARGS!
            if errorlevel 1 exit /b !errorlevel!
        )
    )
)

if "!FOUND!"=="0" (
    echo No non-LAPACK examples found in examples\*.rs 1>&2
    exit /b 1
)

exit /b 0
