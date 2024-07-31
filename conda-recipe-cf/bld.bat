@rem Remember to source the compiler

set MKLROOT=%CONDA_PREFIX%
%PYTHON% setup.py install
if errorlevel 1 exit 1
