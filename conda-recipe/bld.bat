@rem Remember to source the compiler

set MKLROOT=%CONDA_PREFIX%

rem Build wheel package
if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
    %PYTHON% -m pip wheel --no-build-isolation --no-deps .
    if errorlevel 1 exit 1
    copy mkl_random*.whl %WHEELS_OUTPUT_FOLDER%
    if errorlevel 1 exit 1
) ELSE (
    rem Build conda package
    %PYTHON% -m pip install --no-build-isolation --no-deps .
    if errorlevel 1 exit 1
)
