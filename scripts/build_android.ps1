#requires -Version 5.1
<#
.SYNOPSIS
    爱弥斯终端 —— 一键构建 Android Debug APK（Web 资源 → cap sync → Gradle 编译）。

.DESCRIPTION
    把「vite build → npx cap sync android → gradlew assembleDebug」三步串起来，
    并自动注入本机工具链环境变量。工具链路径可用参数覆盖。

.PARAMETER Install
    编译完成后用 adb 安装到已连接设备（等价于 `adb install -r`）。

.PARAMETER Clean
    先执行 `gradlew clean`（彻底重编，耗时更长）。

.PARAMETER JdkHome
    JDK 17 根目录，默认 D:\tools\jdk17\jdk-17.0.21+9。

.PARAMETER AndroidSdk
    Android SDK 根目录，默认 D:\tools\android-sdk。

.PARAMETER GradleUserHome
    Gradle 缓存目录，默认 D:\tools\gradle-home。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_android.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_android.ps1 -Install
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Clean,
    [string]$JdkHome       = 'D:\tools\jdk17\jdk-17.0.21+9',
    [string]$AndroidSdk    = 'D:\tools\android-sdk',
    [string]$GradleUserHome = 'D:\tools\gradle-home'
)

$ErrorActionPreference = 'Stop'

function Write-Step([string]$Text) { Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok([string]$Text)   { Write-Host "  [OK] $Text" -ForegroundColor Green }
function Fail([string]$Text) { Write-Host "`n[FAIL] $Text" -ForegroundColor Red; exit 1 }

$root      = Split-Path -Parent $PSScriptRoot
$androidDir = Join-Path $root 'android'
$apk        = Join-Path $androidDir 'app\build\outputs\apk\debug\app-debug.apk'

Write-Step "环境检查"
if (-not (Test-Path $JdkHome))      { Fail "找不到 JDK：$JdkHome（用 -JdkHome 指定）" }
if (-not (Test-Path $AndroidSdk))   { Fail "找不到 Android SDK：$AndroidSdk（用 -AndroidSdk 指定）" }
if (-not (Test-Path $androidDir))   { Fail "找不到 Android 工程：$androidDir（先 npx cap add android）" }
$env:JAVA_HOME         = $JdkHome
$env:ANDROID_HOME      = $AndroidSdk
$env:ANDROID_SDK_ROOT  = $AndroidSdk
$env:GRADLE_USER_HOME  = $GradleUserHome
$env:PATH              = "$JdkHome\bin;$AndroidSdk\platform-tools;" + $env:PATH
Write-Ok "JAVA_HOME=$JdkHome"
Write-Ok "ANDROID_HOME=$AndroidSdk"
Write-Ok "GRADLE_USER_HOME=$GradleUserHome"

Write-Step "1/3 构建 Web 资源（vite build → dist/）"
Push-Location $root
try {
    & npx vite build
    if ($LASTEXITCODE -ne 0) { Fail "vite build 失败（exit $LASTEXITCODE）" }
    if (-not (Test-Path (Join-Path $root 'dist\index.html'))) { Fail "dist/index.html 未生成" }
    Write-Ok ("dist/index.html {0:N0} B" -f (Get-Item (Join-Path $root 'dist\index.html')).Length)

    Write-Step "2/3 同步进 Android 工程（cap sync android）"
    & npx cap sync android
    if ($LASTEXITCODE -ne 0) { Fail "cap sync 失败（exit $LASTEXITCODE）" }
    Write-Ok "web 资源已同步到 android/app/src/main/assets/public"
}
finally { Pop-Location }

Write-Step "3/3 Gradle 编译（assembleDebug）"
Push-Location $androidDir
try {
    if ($Clean) {
        & .\gradlew.bat clean --no-daemon --console=plain
        if ($LASTEXITCODE -ne 0) { Fail "gradlew clean 失败" }
    }
    & .\gradlew.bat assembleDebug --no-daemon --console=plain
    if ($LASTEXITCODE -ne 0) { Fail "assembleDebug 失败（见上方 Kotlin/AAPT 报错）" }
}
finally { Pop-Location }

if (-not (Test-Path $apk)) { Fail "未找到产物：$apk" }
$info = Get-Item $apk
Write-Host ("`nAPK: {0}`n      {1:N0} B ({2:N2} MB)" -f $info.FullName, $info.Length, ($info.Length / 1MB)) -ForegroundColor Green

if ($Install) {
    Write-Step "安装到设备"
    $adb = Join-Path $AndroidSdk 'platform-tools\adb.exe'
    & $adb devices
    & $adb install -r $apk
    if ($LASTEXITCODE -ne 0) { Fail "adb install 失败（确认已连接真机且开启 USB 调试）" }
    Write-Ok "已安装。日志：adb logcat -s AemeathTtsPlugin AemeathSoVits AemeathSherpa AemeathOrt AemeathDownloader AemeathAudio"
}

Write-Host "`n完成。" -ForegroundColor Cyan
