import uproot
print(f"{'dR':>5} {'mode':>5} {'ECM':>4}  valid_frac")
for tag in ("dr005","dr01","dr02","nocut"):
    for mode in ("pool","swap","fixed"):
        for ecm in (157,160,163):
            f = f"outputs/treemaker/lnuqq/step2_{tag}_{mode}/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
            try:
                v = uproot.open(f)["events"]["kinfit_valid"].array(library="np")
                print(f"{tag:>5} {mode:>5} {ecm:>4}  {100*v.sum()/len(v):6.2f}")
            except Exception:
                print(f"{tag:>5} {mode:>5} {ecm:>4}  --")
