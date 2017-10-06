@rem Remember to source Intel (R) Compiler

set CC=icl
set LD=xilink

%PYTHON% setup.py config --compiler=intelemw install --old-and-unmanageable
if errorlevel 1 exit 1
