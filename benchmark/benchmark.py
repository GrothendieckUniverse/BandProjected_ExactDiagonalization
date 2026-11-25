"""Benchmarks and validation SVG spectra for BandProjected_ExactDiagonalization."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bandprojected_exactdiagonalization import (
    BandProjectedED,
    plot_ed_spectrum,
    thouless_charge_pump_1d,
)
from bandprojected_exactdiagonalization.models import (
    ChiralTDBG,
    IntervalleyTBG,
    P4MMStrainModel,
    TMDContinuum,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS_JSON = ROOT / "benchmark" / "benchmark_results.json"


def _benchmark_mode(ed: BandProjectedED, sector: tuple[int, int], mode: str, nev: int = 3) -> dict:
    bas = ed.catalog.basis(ed.n_particles, sector)
    tracemalloc.start()
    t0 = time.perf_counter()
    if mode == "matrix":
        mat = ed.assemble_sparse_sector(bas)
        t_build = time.perf_counter() - t0
    elif mode == "matrixfree":
        op, _ = ed.make_matrixfree_operator(bas)
        t_build = time.perf_counter() - t0
    else:
        t_build = 0.0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    res = ed.solve_sector(sector, nev=nev, mode=mode)
    _, peak_solve = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": mode,
        "dimension": int(res.dimension),
        "build_seconds": float(t_build),
        "solve_seconds": float(res.elapsed_seconds),
        "peak_memory_mb": float(max(peak_build, peak_solve) / (1024**2)),
        "ground_energy": float(res.energies[0]),
    }


def _benchmark_thread_scaling() -> list[dict]:
    m0 = P4MMStrainModel(alpha=0.5279, beta=0.0, cutoff=10.0)
    _, h0_ph, _, _ = m0.hole_partner_hamiltonian((3, 4), qell_cutoff=16.0)
    rows = []
    for nt in (1, 2, 4, 8, 12):
        ed = BandProjectedED(h0_ph, n_particles=8, num_threads=nt)
        bas = ed.catalog.basis(8, (1, 0))
        v = np.ones(len(bas), dtype=np.complex128)
        pool = ThreadPoolExecutor(max_workers=nt) if nt > 1 else None
        try:
            op, _ = ed.make_matrixfree_operator(bas, pool=pool)
            _ = op @ v  # warm up
            t0 = time.perf_counter()
            for _ in range(5):
                _ = op @ v
            dt = (time.perf_counter() - t0) / 5.0
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        rows.append({"threads": nt, "matvec_seconds": float(dt), "dimension": int(len(bas))})
    return rows


def _benchmark_sector_workers() -> list[dict]:
    model = TMDContinuum(twist_degrees=3.89, dielectric=15.0, cutoff=6)
    _, ham, _, _ = model.hole_partner_hamiltonian(supercell=(4, 6), qell_cutoff=7.5)
    ed = BandProjectedED(ham, n_particles=8, num_threads=1)
    sectors = [(0, 0), (0, 2), (0, 4), (1, 0), (1, 2), (1, 4), (2, 0), (2, 2)]
    rows = []
    for nw in (1, 2, 4, 8):
        t0 = time.perf_counter()
        ed.scan_sectors(sectors=sectors, nev=3, mode="matrix", sector_workers=nw)
        rows.append({"sector_workers": nw, "n_sectors": len(sectors), "seconds": float(time.perf_counter() - t0)})
    return rows


def _plot_tmote2_evolution(scan_ref: dict, scan_edge: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True, constrained_layout=True)
    fci_secs = {(0, 0), (0, 2), (0, 4)}

    # Left panel: gapped arXiv:2304.11864 FCI at theta=3.89 deg, eps=15
    # Right panel: FCI-FL transition at theta=4.0 deg, eps=18 with (2,0) intruder
    configs = (
        (
            axes[0],
            scan_ref,
            r"(a) $t\mathrm{MoTe}_2$ ($\theta=3.89^\circ, \epsilon=15$, $\nu_h=-2/3$)",
            False,
        ),
        (
            axes[1],
            scan_edge,
            r"(b) $t\mathrm{MoTe}_2$ ($\theta=4.0^\circ, \epsilon=18$, FCI–FL transition)",
            True,
        ),
    )

    for ax, scan, title, mark_intruder in configs:
        n1, n2 = scan["shape"]
        e0 = float(scan["ground_energy"])
        plotted_fci = False
        plotted_exc = False
        plotted_int = False
        for s in sorted(scan["sectors"].keys()):
            x = s[0] * n2 + s[1]
            es = scan["sectors"][s].energies - e0
            for lvl, e in enumerate(es):
                if lvl == 0 and s in fci_secs:
                    lbl = "FCI sectors (0,0), (0,2), (0,4)" if not plotted_fci else None
                    plotted_fci = True
                    ax.scatter([x], [e], color="#d95f02", s=50, zorder=5, edgecolors="#222222", linewidths=0.7, label=lbl)
                elif mark_intruder and lvl == 0 and s == (2, 0):
                    lbl = "FL intruder (2,0)" if not plotted_int else None
                    plotted_int = True
                    ax.scatter([x], [e], color="#e7298a", marker="D", s=54, zorder=6, edgecolors="#222222", linewidths=0.7, label=lbl)
                else:
                    lbl = "Excited / competing levels" if not plotted_exc else None
                    plotted_exc = True
                    ax.scatter([x], [e], color="#3b6ea5", s=26, alpha=0.85, zorder=3, label=lbl)

        if not mark_intruder:
            e_top = float(scan["ordered_levels"][2][0] - e0)
            e_exc = float(scan["ordered_levels"][3][0] - e0)
            gap = e_exc - e_top
            ax.axhspan(e_top, e_exc, color="#1b9e77", alpha=0.14, zorder=1)
            ax.axhline(e_top, color="#1b9e77", linestyle=":", linewidth=1.0, zorder=2)
            ax.axhline(e_exc, color="#1b9e77", linestyle="--", linewidth=1.0, zorder=2)
            ax.text(
                0.97,
                0.52,
                rf"$\Delta_{{\mathrm{{FCI}}}} = {gap:.2f}\ \mathrm{{meV}}$" + "\n" + rf"spread $= {e_top:.3f}\ \mathrm{{meV}}$",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1b9e77", alpha=0.9),
            )
        else:
            e00 = float(scan["sectors"][(0, 0)].energies[0] - e0)
            e20 = float(scan["sectors"][(2, 0)].energies[0] - e0)
            ax.annotate(
                rf"(2,0) $={e20:.3f}$ < (0,0) $={e00:.3f}$",
                xy=(2 * n2, e20),
                xytext=(12.2, 0.72),
                fontsize=8.8,
                arrowprops=dict(arrowstyle="->", color="#e7298a", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#e7298a", alpha=0.9),
            )

        ax.set_xlabel(r"Total momentum sector index $k_1 N_2 + k_2$")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8.2)

    axes[0].set_ylabel(r"$E - E_0$ (meV)")
    axes[0].set_ylim(-0.12, 2.85)
    out = FIGURES / "tmote2_ed_spectrum.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def _plot_benchmark_svg(size_rows: list[dict], thread_rows: list[dict], worker_rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)

    # Panel 1: Matrix vs Matrix-free solve time & memory
    ax = axes[0]
    labels = [r["label"] for r in size_rows]
    t_mat = [r["matrix"]["solve_seconds"] for r in size_rows]
    t_mf = [r["matrixfree"]["solve_seconds"] for r in size_rows]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, t_mat, width=w, color="#3b6ea5", label="CSR matrix (eigsh)")
    ax.bar(x + w / 2, t_mf, width=w, color="#d95f02", label="Zero-loop matrix-free (nt=12)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Single-sector solve time (s)")
    ax.set_title("Solver Time vs Hilbert Dimension")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8)

    # Panel 2: Matrix-free thread scaling (p4mm dim=61,333)
    ax = axes[1]
    ths = [r["threads"] for r in thread_rows]
    mvs = [r["matvec_seconds"] * 1000.0 for r in thread_rows]
    ax.plot(ths, mvs, "o-", color="#1b9e77", linewidth=2.0, markersize=6)
    ax.set_xlabel(r"Worker threads (`num_threads`)")
    ax.set_ylabel("Matvec latency (ms)")
    ax.set_title(r"Matrix-Free Scaling ($p4mm$, $D=61{,}333$)")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # Panel 3: Multi-sector process scaling (8 sectors of tMoTe2 4x6, D=30,667)
    ax = axes[2]
    nws = [r["sector_workers"] for r in worker_rows]
    secs = [r["seconds"] for r in worker_rows]
    ax.plot(nws, secs, "s-", color="#7570b3", linewidth=2.0, markersize=6)
    ax.set_xlabel(r"Sector workers (`sector_workers`)")
    ax.set_ylabel("8-sector wall time (s)")
    ax.set_title(r"Process-Safe Sector Scaling ($t\mathrm{MoTe}_2$, $4\times 6$)")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    out = FIGURES / "benchmark_scaling.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def _plot_ctdbg_thouless_pump(t01: dict, t10: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, res, title in (
        (axes[0], t01, r"cTDBG $\beta=0.10$ ($\mathcal{C}=1$, $\mathcal{C}_e=1/3$)"),
        (axes[1], t10, r"cTDBG $\beta=1.00$ ($\mathcal{C}=2$, $\mathcal{C}_e=2/3$)"),
    ):
        phi = res["thetas"] / (2.0 * np.pi)
        phi_closed = np.r_[phi, 1.0]
        ax.plot(phi_closed, res["pumped_cycle"], "o-", color="#d95f02", linewidth=2.0, label="Pumped charge $Q(\\phi_y)$")
        ax.axhline(res["averaged_hall_conductivity"], color="#3b6ea5", linestyle="--", linewidth=1.2, label=rf"$\mathcal{{C}}_e = {res['integer_chern']}/3$")
        ax_r = ax.twinx()
        e0 = min(v[:, 0].min() for v in res["spectral_flow_by_sector"].values())
        for s, vals in sorted(res["spectral_flow_by_sector"].items()):
            ax_r.plot(phi, vals[:, 0] - e0, "--", alpha=0.55, linewidth=1.2, label=f"sector {s}")
        ax.set_xlabel(r"Flux insertion $\phi_y / 2\pi$")
        ax.set_ylabel(r"Multiplet-averaged pumped charge")
        ax_r.set_ylabel(r"$E_0(\phi_y) - E_{0,\min}$ (meV)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="upper left", fontsize=8)
    out = FIGURES / "ctdbg_thouless_pump.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def _plot_p4mm_endpoints(scan0: dict, e_full0: float, scan1: dict, e_full1: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    for ax, scan, efull, title in (
        (axes[0], scan0, e_full0, r"$p4mm$ $\nu=4/3$ ($\alpha=0.5279, \beta=0.0$)"),
        (axes[1], scan1, e_full1, r"$p4mm$ $\nu=4/3$ ($\alpha=0.6651, \beta=0.7183$)"),
    ):
        n1, n2 = scan["shape"]
        e0 = scan["ground_energy"] + efull
        top = {(s, lvl) for _, s, lvl in scan["ordered_levels"][:3]}
        for s in sorted(scan["sectors"].keys()):
            x = s[0] * n2 + s[1]
            es = scan["sectors"][s].energies + efull - e0
            for lvl, e in enumerate(es):
                if (s, lvl) in top:
                    ax.scatter([x], [e], color="#d95f02", s=46, zorder=4, edgecolors="#222222", linewidths=0.6)
                else:
                    ax.scatter([x], [e], color="#3b6ea5", s=26, alpha=0.85, zorder=3)
        ax.set_xlabel(r"Total momentum sector $k_1 N_2 + k_2$")
        ax.set_ylabel(r"$E - E_0$ ($U$)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    out = FIGURES / "p4mm_endpoints_ed_spectrum.svg"
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("=== 1. Solver Mode Benchmarks ===", flush=True)
    size_specs = [
        ("tMoTe2 3x5\n(D=201)", (3, 5), 5, (0, 0)),
        ("tMoTe2 3x6\n(D=1,032)", (3, 6), 6, (0, 0)),
        ("tMoTe2 4x6\n(D=30,709)", (4, 6), 8, (0, 0)),
    ]
    size_rows = []
    for label, sc, ne, sec in size_specs:
        m = TMDContinuum(twist_degrees=3.89, dielectric=15.0, cutoff=5)
        _, h, _, _ = m.hole_partner_hamiltonian(supercell=sc, qell_cutoff=7.0)
        ed_mat = BandProjectedED(h, n_particles=ne, num_threads=12)
        row = {
            "label": label,
            "matrix": _benchmark_mode(ed_mat, sec, "matrix"),
            "matrixfree": _benchmark_mode(ed_mat, sec, "matrixfree"),
        }
        size_rows.append(row)
        print(label.replace("\n", " "), row["matrix"]["solve_seconds"], row["matrixfree"]["solve_seconds"], flush=True)

    m_p4 = P4MMStrainModel(alpha=0.5279, beta=0.0, cutoff=10.0)
    _, h_p4_ph, e_full0, _ = m_p4.hole_partner_hamiltonian((3, 4), qell_cutoff=16.0)
    ed_p4 = BandProjectedED(h_p4_ph, n_particles=8, num_threads=12)
    p4_row = {
        "label": "p4mm 3x4\n(D=61,333)",
        "matrix": _benchmark_mode(ed_p4, (1, 0), "matrix"),
        "matrixfree": _benchmark_mode(ed_p4, (1, 0), "matrixfree"),
    }
    size_rows.append(p4_row)
    print("p4mm 3x4 (D=61,333):", p4_row["matrix"]["solve_seconds"], p4_row["matrixfree"]["solve_seconds"], flush=True)

    print("=== 2. Thread & Sector-Worker Scaling ===", flush=True)
    thread_rows = _benchmark_thread_scaling()
    worker_rows = _benchmark_sector_workers()
    _plot_benchmark_svg(size_rows, thread_rows, worker_rows)

    print("=== 3. Model Full-Spectrum & Charge-Pump Figures ===", flush=True)
    _, h_tm_ref, _, _ = TMDContinuum().hole_partner_hamiltonian((4, 6), qell_cutoff=7.5)
    scan_tm_ref = BandProjectedED(h_tm_ref, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    _, h_tm_edge, _, _ = TMDContinuum(twist_degrees=4.0, dielectric=18.0, cutoff=6).hole_partner_hamiltonian((4, 6), qell_cutoff=7.5)
    scan_tm_edge = BandProjectedED(h_tm_edge, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    _plot_tmote2_evolution(scan_tm_ref, scan_tm_edge)

    _, h_c01 = ChiralTDBG(beta=0.1, cutoff=5.0).hamiltonian((4, 6), qell_cutoff=6.0)
    scan_c01 = BandProjectedED(h_c01, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    plot_ed_spectrum(scan_c01, FIGURES / "ctdbg_beta0p1_ed_spectrum.svg", title=r"Chiral TDBG $\beta=0.10$ ($C=2$, $\mathcal{C}_e=1/3$)", energy_unit="meV")

    _, h_c10 = ChiralTDBG(beta=1.0, cutoff=5.0).hamiltonian((4, 6), qell_cutoff=6.0)
    scan_c10 = BandProjectedED(h_c10, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    plot_ed_spectrum(scan_c10, FIGURES / "ctdbg_beta1p0_ed_spectrum.svg", title=r"Chiral TDBG $\beta=1.00$ ($C=2$, $\mathcal{C}_e=2/3$)", energy_unit="meV")

    pump01 = thouless_charge_pump_1d(
        lambda tw: ChiralTDBG(beta=0.1, cutoff=4.0).hamiltonian((4, 6), qell_cutoff=6.0, twists=tw),
        n_particles=8, n_steps=12, flux_axis=1, manifold_size=3, sectors=[(0, 0), (0, 2), (0, 4)], nev=2, mode="matrix", sector_workers=3,
    )
    pump10 = thouless_charge_pump_1d(
        lambda tw: ChiralTDBG(beta=1.0, cutoff=4.0).hamiltonian((4, 6), qell_cutoff=6.0, twists=tw),
        n_particles=8, n_steps=12, flux_axis=1, manifold_size=3, sectors=[(0, 0), (0, 2), (0, 4)], nev=2, mode="matrix", sector_workers=3,
    )
    _plot_ctdbg_thouless_pump(pump01, pump10)

    _, h_ic = IntervalleyTBG(cutoff=4.0, dielectric=4.0).hamiltonian((4, 6), qell_cutoff=6.0)
    scan_ic = BandProjectedED(h_ic, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    plot_ed_spectrum(scan_ic, FIGURES / "ictbg_ed_spectrum.svg", title=r"Intervalley TBG ($C=0$, $4\times 6$, $\nu=1/3$)", energy_unit="meV")

    scan_p0 = BandProjectedED(h_p4_ph, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    m_p4_1 = P4MMStrainModel(alpha=0.6651, beta=0.7183, cutoff=10.0)
    _, h_p4_1, e_full1, _ = m_p4_1.hole_partner_hamiltonian((3, 4), qell_cutoff=16.0)
    scan_p1 = BandProjectedED(h_p4_1, 8).scan_sectors(nev=4, mode="matrix", sector_workers=12)
    _plot_p4mm_endpoints(scan_p0, e_full0, scan_p1, e_full1)

    summary = {
        "size_scaling": size_rows,
        "thread_scaling": thread_rows,
        "sector_worker_scaling": worker_rows,
        "spectra_gaps": {
            "tmote2_3p89_eps15": float(scan_tm_ref["ordered_levels"][3][0] - scan_tm_ref["ordered_levels"][2][0]),
            "tmote2_4p0_eps18": float(scan_tm_edge["ordered_levels"][3][0] - scan_tm_edge["ordered_levels"][2][0]),
            "ctdbg_beta0p1": float(scan_c01["ordered_levels"][3][0] - scan_c01["ordered_levels"][2][0]),
            "ctdbg_beta1p0": float(scan_c10["ordered_levels"][3][0] - scan_c10["ordered_levels"][2][0]),
            "ictbg": float(scan_ic["ordered_levels"][3][0] - scan_ic["ordered_levels"][2][0]),
            "p4mm_0": float(scan_p0["ordered_levels"][3][0] - scan_p0["ordered_levels"][2][0]),
            "p4mm_1": float(scan_p1["ordered_levels"][3][0] - scan_p1["ordered_levels"][2][0]),
        },
        "ctdbg_pumps": {
            "beta0p1_chern": int(pump01["integer_chern"]),
            "beta1p0_chern": int(pump10["integer_chern"]),
        },
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2))
    print("Benchmark & SVG figure suite completed cleanly.", flush=True)


if __name__ == "__main__":
    main()
