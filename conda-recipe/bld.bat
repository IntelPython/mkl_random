:: -wnx flags mean: --wheel --no-isolation --skip-dependency-check
:: -Ccompile-args=-v makes ninja print full compiler commands (verbose build)
%PYTHON% -m build -w -n -x -Ccompile-args=-v
if errorlevel 1 exit 1

for /f %%f in ('dir /b /S .\dist') do (
  %PYTHON% -m pip install %%f ^
    --no-build-isolation ^
    --no-deps ^
    --only-binary :all: ^
    --no-index ^
    --prefix %PREFIX% ^
    -vv
  if errorlevel 1 exit 1
)

if NOT "%WHEELS_OUTPUT_FOLDER%"=="" (
  copy dist\mkl_random*.whl %WHEELS_OUTPUT_FOLDER%
  if errorlevel 1 exit 1
)
