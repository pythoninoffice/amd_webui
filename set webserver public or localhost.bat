@echo off

title amd_webui webserver public or localhost by SimolZimol
set PYTHON=python
set networkfile =venv/Lib/site-packages/gradio/networking.py

if exist tmp/webset.txt (
	goto st1
) else (
	%PYTHON% -c '' >tmp/webset.txt
)
:st1
for /f %%i in ("tmp/webset.txt") do set size=%%~zi
if %size% gtr 0 goto file

echo do you want to make the website public or localhost (default) ?
echo localhost = 1 // public = 2
set /p setdata= : 
echo %setdata% >> tmp/webset.txt

if %setdata% == 2 (

Powershell -Command "(Get-Content venv/Lib/site-packages/gradio/networking.py).replace('127.0.0.1', '0.0.0.0') | Set-Content venv/Lib/site-packages/gradio/networking.py"
)
exit /b
)

:file
set/p Data= < tmp/webset.txt
echo do you want to make the website public or localhost (default) ?
echo localhost = 1 // public = 2
echo The current setting is %Data%
set /p setdata= : 
%PYTHON% -c '' >tmp/webset.txt
echo %setdata% >> tmp/webset.txt

if %setdata% == %Data% (
) else (
	if %setdata% == 1 (
		Powershell -Command "(Get-Content venv/Lib/site-packages/gradio/networking.py).replace('0.0.0.0', '127.0.0.1') | Set-Content venv/Lib/site-packages/gradio/networking.py"
	) else (
		Powershell -Command "(Get-Content venv/Lib/site-packages/gradio/networking.py).replace('127.0.0.1', '0.0.0.0') | Set-Content venv/Lib/site-packages/gradio/networking.py"
	)
	
)
exit /b

:: Made by SimolZimol#5242