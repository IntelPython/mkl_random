:: MKL is found through %LIBRARY_PREFIX%\lib\cmake\mkl, which is on
:: CMAKE_PREFIX_PATH via the vs2022 activation script. Setting MKLROOT here
:: would break the build: MKLConfig.cmake looks for %MKLROOT%\include with
:: NO_DEFAULT_PATH and %PREFIX%\include holds no MKL headers.
%PYTHON% -m pip install --no-build-isolation --no-deps .
if errorlevel 1 exit 1
