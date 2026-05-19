from step2point.io import EDM4hepRootReader

reader = EDM4hepRootReader(
    "../edm4hep_frombenchmark/sim-E1261AT600AP180-180_file_14.edm4hep.root",
    collections=(
        "ECalBarrelSiHitsEven",
        "ECalBarrelSiHitsOdd",
    ),
    # shower_limit=1,
)


# i think it does not work properly for ILD because of the even/odd layers in the ECal
shower = next(reader.iter_showers())
print([attr for attr in dir(shower) if not attr.startswith("_")])
print(shower.n_points, shower.cell_id is not None, shower.t is not None)

print("n_points:", shower.n_points)
print("energy sum:", shower.E.sum() if shower.E is not None else "N/A")
print("cell_id sample:", shower.cell_id[:5] if shower.cell_id is not None else "N/A")
print("x range:", shower.x.min(), shower.x.max())
print("y range:", shower.y.min(), shower.y.max())
print("z range:", shower.z.min(), shower.z.max())
