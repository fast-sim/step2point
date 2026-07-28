import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot

Y_MAX_BARREL = 2028  # mm
Y_MIN_BARREL = 1804.7  # mm
CONTAINMENT_THRESHOLD = 0.95

seed_list = np.concatenate([np.arange(1, 11), np.arange(30, 45)])
for seed in seed_list:
    ROOT_FILE = f"../photons_root/p22_th45-135_ph79-109_en5-130_seed{seed}_ip.edm4hep.root"
    # OUT_FILE = "/eos/user/m/mamozzan/photons_root_cut/events_barrel_contained.root"

    # ── 1. load ───────────────────────────────────────────────────────────────────
    file = uproot.open(ROOT_FILE)
    tree = file["events"]

    # ── 2. MC particles (per-event theta) ─────────────────────────────────────────
    gen_status = tree["MCParticles/MCParticles.generatorStatus"].array()
    mc_mask = gen_status == 1

    px_mc = tree["MCParticles/MCParticles.momentum.x"].array()[mc_mask]
    py_mc = tree["MCParticles/MCParticles.momentum.y"].array()[mc_mask]
    pz_mc = tree["MCParticles/MCParticles.momentum.z"].array()[mc_mask]

    # flatten to 1D for numpy
    px_flat = ak.to_numpy(ak.flatten(px_mc))
    py_flat = ak.to_numpy(ak.flatten(py_mc))
    pz_flat = ak.to_numpy(ak.flatten(pz_mc))
    p_flat = np.sqrt(px_flat**2 + py_flat**2 + pz_flat**2)
    theta_flat = np.degrees(np.arccos(np.clip(pz_flat / p_flat, -1, 1)))

    # one theta per event (first primary particle)
    px_ev = ak.to_numpy(ak.firsts(px_mc))
    py_ev = ak.to_numpy(ak.firsts(py_mc))
    pz_ev = ak.to_numpy(ak.firsts(pz_mc))
    p_ev = np.sqrt(px_ev**2 + py_ev**2 + pz_ev**2)
    theta_per_event = np.degrees(np.arccos(np.clip(pz_ev / p_ev, -1, 1)))

    # ── 3. ECAL hits ──────────────────────────────────────────────────────────────
    def load_hits(tree, collection):
        x = tree[f"{collection}/{collection}.position.x"].array()
        y = tree[f"{collection}/{collection}.position.y"].array()
        z = tree[f"{collection}/{collection}.position.z"].array()
        E = tree[f"{collection}/{collection}.energy"].array()
        return x, y, z, E

    x_e, y_e, z_e, E_e = load_hits(tree, "ECalBarrelSiHitsEven")
    x_o, y_o, z_o, E_o = load_hits(tree, "ECalBarrelSiHitsOdd")

    x_hits = ak.concatenate([x_e, x_o], axis=1)
    y_hits = ak.concatenate([y_e, y_o], axis=1)
    z_hits = ak.concatenate([z_e, z_o], axis=1)
    E_hits = ak.concatenate([E_e, E_o], axis=1)
    print(ak.min(x_hits), ak.max(x_hits))

    # ── 4. per-event barrel containment ──────────────────────────────────────────
    barrel_mask = np.abs(y_hits) < Y_MAX_BARREL
    E_barrel = ak.sum(E_hits[barrel_mask], axis=1)
    E_total = ak.sum(E_hits, axis=1)
    containment = ak.to_numpy(ak.where(E_total > 0, E_barrel / E_total, 1.0))

    # ── 5. find worst (most leaking) event ───────────────────────────────────────
    worst_idx = int(np.argmin(containment))
    print(f"Most leaking event : {worst_idx}")
    print(f"Barrel containment : {containment[worst_idx]:.3f}")
    print(f"Theta (MC)         : {theta_per_event[worst_idx]:.2f} deg")

    x_w = ak.to_numpy(x_hits[worst_idx])
    y_w = ak.to_numpy(y_hits[worst_idx])
    z_w = ak.to_numpy(z_hits[worst_idx])
    E_w = ak.to_numpy(E_hits[worst_idx])
    r_w = np.sqrt(x_w**2 + y_w**2)

    in_barrel = np.abs(y_w) < Y_MAX_BARREL
    in_endcap = np.abs(y_w) >= Y_MAX_BARREL

    # ── 6. plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Most endcap-leaking shower  (event {worst_idx},  "
        f"barrel containment = {containment[worst_idx]:.1%},  "
        f"θ = {theta_per_event[worst_idx]:.1f}°)",
        fontsize=13,
    )

    kw_b = dict(c="steelblue", alpha=0.6, label="Barrel")
    kw_e = dict(c="tomato", alpha=0.6, label="Endcap")

    ax = axes[0]
    ax.scatter(y_w[in_barrel], r_w[in_barrel], s=E_w[in_barrel] * 500, **kw_b)
    ax.scatter(y_w[in_endcap], r_w[in_endcap], s=E_w[in_endcap] * 500, **kw_e)
    ax.axvline(Y_MAX_BARREL, color="k", ls="--", label=f"|z| = {Y_MAX_BARREL} mm")
    ax.axvline(Y_MIN_BARREL, color="k", ls="--")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Side view (r vs y)")
    ax.legend()

    ax = axes[1]
    ax.scatter(x_w[in_barrel], z_w[in_barrel], s=E_w[in_barrel] * 500, **kw_b)
    ax.scatter(x_w[in_endcap], z_w[in_endcap], s=E_w[in_endcap] * 500, **kw_e)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Transverse view (x vs z)")
    ax.legend()

    ax = axes[2]
    ax.scatter(y_w[in_barrel], E_w[in_barrel], s=10, **kw_b)
    ax.scatter(y_w[in_endcap], E_w[in_endcap], s=10, **kw_e)
    ax.axvline(Y_MAX_BARREL, color="k", ls="--", label=f"|z| = {Y_MAX_BARREL} mm")
    ax.axvline(Y_MIN_BARREL, color="k", ls="--")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("E [GeV]")
    ax.set_title("Energy vs y")
    ax.legend()

    plt.tight_layout()
    plt.savefig("martina_test/endcap_leaking_shower.png", dpi=150)
    plt.show()
    print("Saved endcap_leaking_shower.png")

    # ── 7. apply cut and write new ROOT file ──────────────────────────────────────
    event_cut = containment >= CONTAINMENT_THRESHOLD
    print(f"\nEvents passing {CONTAINMENT_THRESHOLD:.0%} barrel containment cut: {event_cut.sum()} / {len(event_cut)}")

# with uproot.recreate(OUT_FILE) as out_file:
#     data = {}
#     for branch in tree.keys():
#         try:
#             data[branch] = tree[branch].array()[event_cut]
#         except Exception as e:
#             print(f"  skipping {branch}: {e}")
#     out_file["events"] = data

# print(f"Written to {OUT_FILE}")
