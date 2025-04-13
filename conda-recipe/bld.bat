@rem Remember to source the compiler

set MKLROOT=%CONDA_PREFIX%
%PYTHON% -m pip install --no-build-isolation --no-deps .
if errorlevel 1 exit 1
