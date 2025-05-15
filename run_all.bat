@echo off
echo digital_image_env
conda activate digital_image_env
echo imread1.py...
python imread1.py
echo.
echo TMO_MSR.py...
python TMO_MSR.py
echo.
echo TMO_MSR_DPHE.py...
python TMO_MSR_DPHE.py
echo.
echo DONE！
pause