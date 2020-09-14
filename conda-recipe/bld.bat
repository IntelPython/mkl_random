@rem Remember to source the compiler

set MKLROOT=%CONDA_PREFIX%
%PYTHON% setup.py install --old-and-unmanageable
if errorlevel 1 exit 1
