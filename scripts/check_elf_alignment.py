"""检查 .so 的 PT_LOAD 对齐（Android 15+ 16KB page size 要求 p_align >= 0x4000）。"""
import struct
import sys
import zipfile
import io

PT_LOAD = 1


def load_aligns(data: bytes):
    if data[:4] != b"\x7fELF":
        return None
    is64 = data[4] == 2
    little = data[5] == 1
    end = "<" if little else ">"
    if is64:
        e_phoff = struct.unpack_from(end + "Q", data, 32)[0]
        e_phentsize = struct.unpack_from(end + "H", data, 54)[0]
        e_phnum = struct.unpack_from(end + "H", data, 56)[0]
    else:
        e_phoff = struct.unpack_from(end + "I", data, 28)[0]
        e_phentsize = struct.unpack_from(end + "H", data, 42)[0]
        e_phnum = struct.unpack_from(end + "H", data, 44)[0]
    aligns = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack_from(end + "I", data, off)[0]
        if p_type == PT_LOAD:
            if is64:
                p_align = struct.unpack_from(end + "Q", data, off + 48)[0]
            else:
                p_align = struct.unpack_from(end + "I", data, off + 28)[0]
            aligns.append(p_align)
    return aligns


def check_aar(path: str):
    print(f"== {path}")
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".so")]
        if not names:
            print("  (no .so)")
            return
        worst = 1 << 62
        for name in sorted(names):
            data = zf.read(name)
            aligns = load_aligns(data)
            if aligns is None:
                print(f"  {name}: not ELF")
                continue
            mn = min(aligns)
            worst = min(worst, mn)
            flag = "OK" if mn >= 0x4000 else "UNALIGNED"
            print(f"  {name}: p_align={[hex(a) for a in aligns]} min={hex(mn)} {flag}")
        print(f"  => worst alignment {hex(worst)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check_aar(p)
