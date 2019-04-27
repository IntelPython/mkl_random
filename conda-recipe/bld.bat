@rem Remember to source the compiler

%PYTHON% setup.py install --old-and-unmanageable
if errorlevel 1 exit 1
