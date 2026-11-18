@echo off
chcp 65001 >nul
echo ========================================================
echo       爱弥斯 (Aemeath) 终端 - 一键推送代码至 GitHub
echo ========================================================
echo.
set /p REPO_URL="请输入您的 GitHub 仓库地址 (例如 https://github.com/YourName/Aemeath.git): "
if "%REPO_URL%"=="" (
    echo [错误] 仓库地址不能为空！
    pause
    exit /b 1
)

cd /d "%~dp0"
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo.
echo 正在推送到 GitHub... (如果是首次推送，Windows 将自动弹出 GitHub 网页登录授权)
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo             代码已成功推送至 GitHub！
    echo ========================================================
) else (
    echo.
    echo [提示] 推送遇到问题，请检查：
    echo 1. GitHub 仓库地址是否正确
    echo 2. 是否已在弹出的网页中授权登录 GitHub
    echo 3. 本地网络是否能够顺畅访问 github.com
)
echo.
pause
