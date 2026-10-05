#!/usr/bin/env python
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORGANS = [
    ("lung_minus_gtv", "Lung (minus GTV)", "#1f4e79", 2.6),
    ("heart",          "Heart",            "#b00020", 1.6),
    ("esophagus",      "Esophagus",        "#2e7d32", 1.6),
    ("spinal_cord",    "Spinal cord",      "#7a5195", 1.6),
    ("gtv",            "GTV",              "#e8820c", 1.8),
    ("ptv",            "PTV",              "#f0a500", 1.6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="path to a DVH/dvh.json")
    ap.add_argument("--out", default="dvh_example.png")
    args = ap.parse_args()
    d = json.load(open(args.json))
    grid_gy = [g / 100.0 for g in d["dose_grid_cGy"]]
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lung = d["structures"].get("lung_minus_gtv") or d["structures"].get("lung")
    for key, lab, col, lw in ORGANS:
        st = d["structures"].get(key)
        if not st or "dvh_pct" not in st:
            continue
        ax.plot(grid_gy, st["dvh_pct"], color=col, lw=lw, label=lab)
    if lung:
        for d_gy, key in [(5, "V5"), (20, "V20")]:
            v = lung.get(key)
            if v is None:
                continue
            ax.plot([d_gy, d_gy], [0, v], ls=":", color="#1f4e79", lw=1.0)
            ax.plot([0, d_gy], [v, v], ls=":", color="#1f4e79", lw=1.0)
            ax.annotate(f"$V_{{{d_gy}}}$={v:.0f}%", (d_gy, v), xytext=(d_gy + 2, v + 4),
                        fontsize=10, color="#1f4e79")
        mld = lung.get("Dmean_cGy")
        if mld:
            ax.axvline(mld / 100.0, color="0.5", ls="--", lw=1.0)
            ax.annotate(f"MLD {mld/100:.1f} Gy", (mld / 100.0, 92),
                        xytext=(mld / 100.0 + 1.5, 92), fontsize=9, color="0.4")
    ax.set_xlabel("Dose (Gy)", fontsize=12); ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_ylim(0, 101); ax.set_xlim(0, max(grid_gy)); ax.grid(alpha=0.3)
    ax.legend(fontsize=10, frameon=False)
    ax.set_title(f"Cumulative DVH - patient {d.get('uuid','')}", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
