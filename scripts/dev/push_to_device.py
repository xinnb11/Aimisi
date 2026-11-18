import subprocess
import sys
from pathlib import Path

ADB = r"D:\tools\android-sdk\platform-tools\adb.exe"
APK = r"D:\语音模型项目\Aemeath-爱弥斯终端-v2.7-debug.apk"
PUBLISH = Path(r"D:\爱弥斯\_work\publish")

def run(cmd):
    print(f">> {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print("ERR:", res.stderr.strip())
    return res.returncode

def main():
    devices = subprocess.check_output([ADB, "devices"], text=True)
    print(devices)
    lines = [l.strip() for l in devices.strip().split("\n")[1:] if l.strip()]
    if not lines or not any("device" in l for l in lines):
        print("No adb device connected!")
        return 1
    
    print("1. Installing APK...")
    run(f'"{ADB}" install -r "{APK}"')
    
    print("2. Pushing conditioning files...")
    run(f'"{ADB}" shell mkdir -p /sdcard/Download/aemeath_update')
    run(f'"{ADB}" push "{PUBLISH}/manifest.json" /sdcard/Download/aemeath_update/')
    run(f'"{ADB}" push "{PUBLISH}/conditioning" /sdcard/Download/aemeath_update/')
    
    target = "/data/user/0/com.aemeath.terminal/files/models/aemeath/aemeath-sovits-pro"
    cp_cmd = f'run-as com.aemeath.terminal cp -r /sdcard/Download/aemeath_update/* {target}/'
    print(f"Copying to sandbox: {cp_cmd}")
    run(f'"{ADB}" shell "{cp_cmd}"')
    
    run(f'"{ADB}" shell rm -rf /sdcard/Download/aemeath_update')
    print("All updates pushed to device successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
