#!/usr/bin/env python3
"""Write the Gumbel and score ROMs that the RTL $readmemh's, from the golden model.

    python3 scripts/gen_roms.py      -> src/rtl/rom/gumbel_rom.hex, src/rtl/rom/score_rom.hex
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "golden"))
import wm_ref  # noqa: E402

out = ROOT / "src" / "rtl" / "rom"
out.mkdir(parents=True, exist_ok=True)
for name, rom in (("gumbel_rom.hex", wm_ref.G_ROM), ("score_rom.hex", wm_ref.S_ROM)):
    words = [f"{int(x) & 0xFFFF:04x}" for x in rom]
    (out / name).write_text("\n".join(words) + "\n")
    print(f"{name}: {len(words)} x 16 bit")
