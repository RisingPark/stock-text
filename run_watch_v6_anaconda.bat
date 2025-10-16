@echo off
echo Starting KRX Stock Price Monitor v6 (Anaconda)...
echo ======================================
echo.

REM Use Anaconda Python directly
set PYTHON_PATH=Z:\anaconda3\python.exe
set PIP_PATH=Z:\anaconda3\Scripts\pip.exe

REM Check if Anaconda Python exists
if not exist "%PYTHON_PATH%" (
    echo Error: Anaconda Python not found at %PYTHON_PATH%
    echo Please check your Anaconda installation path
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%
"%PYTHON_PATH%" --version
echo.

REM Install required packages if needed
echo Checking required packages...

"%PYTHON_PATH%" -c "import pykrx" 2>nul
if errorlevel 1 (
    echo Installing pykrx...
    "%PIP_PATH%" install pykrx
)

"%PYTHON_PATH%" -c "import pandas" 2>nul
if errorlevel 1 (
    echo Installing pandas...
    "%PIP_PATH%" install pandas
)

"%PYTHON_PATH%" -c "import numpy" 2>nul
if errorlevel 1 (
    echo Installing numpy...
    "%PIP_PATH%" install numpy
)

"%PYTHON_PATH%" -c "import psutil" 2>nul
if errorlevel 1 (
    echo Installing psutil...
    "%PIP_PATH%" install psutil
)

echo.
echo Starting monitor...
echo Press Ctrl+C to stop
echo ======================================
echo.

REM Run the script with Anaconda Python
"%PYTHON_PATH%" watch_writer_v6_krx.py --interval 10

pause