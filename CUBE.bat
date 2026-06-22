@echo off
powershell.exe -ExecutionPolicy ByPass -NoProfile -Command "& 'C:\Users\param\anaconda3\shell\condabin\conda-hook.ps1'; conda activate cube; cd 'D:\CUBE'; python cube.py"
