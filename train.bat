:: first change dir to where this batch file is placed
cd /D "%~dp0"

:: then start the python server and the game client
start .\server\Scripts\activate ^&^& .\train.py
