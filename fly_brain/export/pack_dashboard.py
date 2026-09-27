"""Repack dashboard/data binaries into web-servable files: activity -> 8-bit grayscale PNG, xyz/region -> base64 .js."""
import base64, json, struct, zlib
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "dashboard" / "data"
W = 376

def png_gray(a):  # a: uint8 (H, W)
    h, w = a.shape
    raw = b"".join(b"\x00" + a[i].tobytes() for i in range(h))
    def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")

idx = json.loads((D / "index.json").read_text(encoding="utf-8"))
nP, nB = idx["points"], idx["bins"]
H = -(-nP // W)  # rows per bin
xyz = (D / "brain_xyz.bin").read_bytes(); reg = (D / "point_region.bin").read_bytes()
(D / "brain.js").write_text("window.FLY_XYZ=" + json.dumps(base64.b64encode(xyz).decode()) + ";window.FLY_REGION=" + json.dumps(base64.b64encode(reg).decode()) + ";")
idx["png"] = {"width": W, "rowsPerBin": H}
for d in idx["drinks"]:
    a = np.fromfile(D / d["file"], dtype=np.uint8).reshape(nB, nP)
    img = np.zeros((nB, H * W), np.uint8); img[:, :nP] = a
    png = png_gray(img.reshape(nB * H, W))
    d["png"] = d["id"] + ".png"; (D / d["png"]).write_bytes(png)
    print(d["id"], len(png) // 1024, "KB")
(D / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
print("brain.js", (D / "brain.js").stat().st_size // 1024, "KB")
