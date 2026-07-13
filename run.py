# run.py — main entry, thread pool scheduling, progress bar, component distribution statistics report

import os, sys
import psutil
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from core import CK, LG, ML, RT, DR, mok
from state import csm, lj, Ss
from pipe import wk

# ═══════════════════════════════════════════════════════════════
# Config: BIRD
# ═══════════════════════════════════════════════════════════════
BIRD_CONFIG = {
    "name": "BIRD",
    "tables_json": os.path.join(RT, '..', 'train_tables.json'),
    "databases_dir": os.path.join(RT, '..', 'train_databases'),
    "seeds_json": os.path.join(RT, '..', 'train_bird_complexity.json'),
    "TR": {
        "moderate": {"range": (5, 7), "seed_max": 5, "target": 1156, "output": "aug_v8_moderate_5_7.json"},
    },
    "raw_counts": {
        'A_JOIN': 2261, 'B_Function': 3024, 'B_Logic': 2065, 'B_Arithmetic': 1195,
        'B_CASE': 452,  'B_Predicate': 593, 'C_GROUP': 221,  'C_HAVING': 53,
        'D_Subquery': 225, 'E_ORDER': 0, 'E_LIMIT': 0, 'E_TOP': 0,
        'F_Window': 2, 'G_SetOp': 8, 'H_DISTINCT': 258,
    },
    "seed_field": "SQL",
    "simple_threshold": 4,
}

# ═══════════════════════════════════════════════════════════════
# Config: Spider
# ═══════════════════════════════════════════════════════════════
SPIDER_CONFIG = {
    "name": "Spider",
    "tables_json": os.path.join(RT, '..', 'tables.json'),
    "databases_dir": os.path.join(RT, '..', 'database'),
    "seeds_json": os.path.join(RT, '..', 'train_spider.json'),
    "TR": {
        "moderate_low":  {"range": (3, 4), "seed_max": 2, "target": 500,
                          "output": "aug_v8_spider_moderate_3_4.json"},
        "moderate_high": {"range": (5, 7), "seed_max": 2, "target": 180,
                          "output": "aug_v8_spider_moderate_5_7.json"},
    },
    "raw_counts": None,
    "seed_field": "query",
    "simple_threshold": 2,
}


def build_targets(raw):
    """Rescale non-E to 99%, assign E_ORDER 0.5% + E_LIMIT 0.5% = 1%."""
    non_e = sum(v for k, v in raw.items() if not k.startswith('E_'))
    tgt = {}
    for k, v in raw.items():
        if k.startswith('E_'): continue
        tgt[k] = (v / non_e) * 0.99 if non_e else 0
    tgt['E_ORDER'] = 0.005; tgt['E_LIMIT'] = 0.005; tgt['E_TOP'] = 0.000
    s = sum(tgt.values())
    return {k: v / s for k, v in tgt.items()}


def run_pipeline(cfg):
    name = cfg["name"]
    print("=" * 60)
    print(f"  V8 SQL Augmentation Pipeline - {name}")
    for t, cf in cfg["TR"].items():
        print(f"  {t}: target={cf['target']}, range={cf['range']}")
    print("=" * 60)

    LG.info("Loading %s schemas and seeds...", name)
    sm = csm(cfg["tables_json"], cfg["databases_dir"])
    sd = lj(cfg["seeds_json"])
    LG.info("Loaded %d schemas, %d seeds", len(sm), len(sd))
    print(f"[MAIN] Schemas={len(sm)}, Seeds={len(sd)}")
    if not sm:
        print("[ERROR] No schemas loaded, exiting")
        sys.exit(1)

    # Build COMP_TARGETS
    if cfg["raw_counts"]:
        comp_targets = {t: build_targets(cfg["raw_counts"]) for t in cfg["TR"]}
    else:
        comp_targets = None

    # Wrap config for state
    class RunConfig:
        pass
    run_cfg = RunConfig()
    run_cfg.TR = cfg["TR"]
    run_cfg.COMP_TARGETS = comp_targets
    run_cfg.seed_field = cfg["seed_field"]
    run_cfg.simple_threshold = cfg["simple_threshold"]

    ttg = sum(cfg["TR"][t]['target'] for t in cfg["TR"])
    pbs = {}
    for i, (t, cf) in enumerate(cfg["TR"].items()):
        lo, hi = cf['range']
        pbs[t] = tqdm(total=cf['target'], desc=f" {t} [{lo}-{hi}]", position=i, leave=True, ncols=80)

    ks = Ss(sd, sm, pbs, run_cfg)
    mw = 16
    LG.info("Workers=%d  TotalTarget=%d  MemLimit=%dGB", mw, ttg, ML)
    print(f"[MAIN] Workers={mw}")
    for t in cfg["TR"]:
        print(f"  {t}: {ks.ccn(t)}/{cfg['TR'][t]['target']}")

    oc, fc, wc = 0, 0, 0
    try:
        with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="W") as pl:
            fts = {}
            rm = sum(cfg["TR"][t]['target'] - ks.ccn(t) for t in cfg["TR"])
            for _ in range(min(mw, rm)):
                if not mok(): break
                tr = ks.pkt()
                if tr is None: break
                fts[pl.submit(wk, wc, ks, tr)] = wc; wc += 1

            while fts and not ks.azc():
                dn, _ = wait(fts.keys(), return_when=FIRST_COMPLETED, timeout=60)
                if not dn:
                    LG.warning("No tasks completed in 60s, pending=%d", len(fts))
                    continue
                for f in dn:
                    w = fts.pop(f, -1)
                    try:
                        if f.result(timeout=5): oc += 1
                        else: fc += 1
                    except Exception as e:
                        fc += 1; LG.error("W-%03d exception: %s", w, e)
                    ta = oc + fc
                    if ta % 20 == 0:
                        rg = psutil.Process().memory_info().rss / 1024**3
                        LG.info("STATUS ok=%d fail=%d | RSS=%.1fGB", oc, fc, rg)
                while len(fts) < mw and not ks.azc():
                    if not mok(): LG.warning("Memory pressure, pausing spawn"); break
                    tr = ks.pkt()
                    if tr is None: break
                    fts[pl.submit(wk, wc, ks, tr)] = wc; wc += 1
            for f in fts: f.cancel()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted - shutting down...")
        LG.warning("KeyboardInterrupt")
    except Exception as e:
        LG.error("MAIN crash: %s", e, exc_info=True)

    for pb in pbs.values(): pb.close()
    LG.info("=" * 70)
    LG.info("FINAL COMPONENT DISTRIBUTION")
    for tn in cfg["TR"]:
        tc = max(1, ks.tcp[tn])
        LG.info("-- %s (%d samples) --", tn, ks.ccn(tn))
        LG.info("  %-14s %8s %8s", "Component", "Actual%", "Count")
        for k in CK:
            LG.info("  %-14s %7.2f%% %7d", k, ks.cpc[tn][k] / tc * 100, ks.cpc[tn][k])
    LG.info("=" * 70)
    mg = f"DONE: ok={oc}, fail={fc}, workers={wc}"
    LG.info(mg)
    print(f"\n[MAIN] {mg}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("bird", "spider"):
        print("Usage: python run.py bird|spider")
        sys.exit(1)

    cfg = BIRD_CONFIG if sys.argv[1] == "bird" else SPIDER_CONFIG
    # Override log name
    global LG
    from core import LP
    run_pipeline(cfg)
