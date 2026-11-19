"""Keep the signed RMSNorm arithmetic and prepend one TP16 reduction block for each row."""

from pathlib import Path
from hashlib import sha256
import json, re, subprocess, sys
from kimi31_residual_stripes_build import device_body
from debug_sources import pin_debug_sources

root = Path(sys.argv[1])
ptx = root / "reduction.ptx"
inv = Path(sys.argv[2])
src = (root / "canonical.ptx").read_text()
metadata = json.loads((root / "canonical.json").read_text())
assert metadata[
    "code_sha256"
] == "2180edc46e524f7280f287c004adbc08592243dd58bb34c300368d7f4692daca" and metadata["block"] == [256, 1, 1]
subprocess.run(
    [
        "/usr/local/cuda-13.2/bin/nvcc",
        "-std=c++17",
        "-O3",
        "-arch=sm_120a",
        "--frandom-seed=31538",
        "-ccbin",
        "/usr/bin/g++-13",
        "--ptx",
        "-rdc=true",
        str(inv),
        "-o",
        str(ptx),
    ],
    check=True,
)
for mode, fun in [("buffer", "row_to_buffer"), ("bank", "row_to_mixture")]:
    code = src.replace(".version 9.3", ".version 9.2").replace(
        ".visible .entry _rms_norm(", f".visible .entry reduce_norm_{mode}("
    )
    code, n = re.subn(
        r"\n\)\n(\.reqntid 256\n\{)", r",\n\t.param .align 8 .b8 exchange_cfg[120]\n)\n\1", code, count=1
    )
    assert n == 1
    if mode == "bank":
        pat = r"ld\.param\.[bu]64\s+(%rd\d+),\s*\[_rms_norm_param_0\];"
        matches = re.findall(pat, code)
        assert len(matches) == 1
        # The final export buffer remains untouched in this component variant.
        # Its canonical input base is now the ready peer bank's full-row view.
        stmt = """
        {
          .reg .u32 lane,row,parity;
          .reg .u64 base,at,delta,off,stride;
          ld.param.u32 lane,[exchange_cfg+112];
          mul.wide.u32 at,lane,8;
          mov.u64 base,exchange_cfg;
          add.u64 at,base,at;
          ld.param.u64 base,[at];
          mov.u32 row,%ctaid.x;
          mul.wide.u32 at,row,4;
          add.u64 at,base,at;
          ld.relaxed.gpu.global.u32 parity,[at];
          and.b32 parity,parity,1;
          ld.param.u64 off,[exchange_cfg+96];
          ld.param.u64 stride,[exchange_cfg+104];
          shl.b64 stride,stride,4;
          cvt.u64.u32 delta,parity;
          mad.lo.u64 at,stride,delta,off;
          add.u64 BASE_REG,base,at;
        }""".replace("BASE_REG", matches[0])
        code = re.sub(pat, stmt, code, count=1)
    start = re.search(r"\.reqntid 256\n\{", code).end()
    body = device_body(ptx.read_text(), fun, 120)
    guard = "{ .reg .u32 row; .reg .pred inactive; mov.u32 row,%ctaid.x; setp.ge.u32 inactive,row,4; @inactive ret; }"
    code = code[:start] + "\n" + body + guard + code[start:]
    path = root / f"fused-{mode}.ptx"
    path.write_text(pin_debug_sources(code))
    subprocess.run(
        [
            "/usr/local/cuda-13.2/bin/ptxas",
            "-O3",
            "-arch=sm_120a",
            str(path),
            "-o",
            str(root / f"fused-{mode}.cubin"),
        ],
        check=True,
    )
    print(mode, sha256((root / f"fused-{mode}.cubin").read_bytes()).hexdigest())
