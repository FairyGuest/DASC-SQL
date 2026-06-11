# run.py — main entry, thread pool scheduling, progress bar, component distribution statistics report

import os, sys
import psutil
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from core import TR, LG, ML, RT, LP, CK, mok
from state import csm, lj, Ss
from pipe import wk

if __name__ == "__main__":
    print("═" * 60)
    print("  V8 SQL Augmentation Pipeline — Spider [3-4] + [5-7]")
    for t, cf in TR.items():
        print(f"  {t}: target={cf['target']}, range={cf['range']}")
    print(f"  Log: {LP}")
    print("═" * 60)

    LG.info("Loading Spider schemas and seeds...")
    tjp = os.path.join(RT, '..', 'tables.json')
    drp = os.path.join(RT, '..', 'database')
    sjp = os.path.join(RT, '..', 'train_spider.json')

    sm = csm(tjp, drp); sd = lj(sjp)
    LG.info("Loaded %d schemas, %d seeds", len(sm), len(sd))
    print(f"[MAIN] Schemas={len(sm)}, Seeds={len(sd)}")
    if not sm: print("[ERROR] No schemas loaded, exiting"); sys.exit(1)

    ttg = sum(TR[t]['target'] for t in TR)
    pbs = {}
    for i, (t, cf) in enumerate(TR.items()):
        lo, hi = cf['range']
        pbs[t] = tqdm(total=cf['target'], desc=f" {t} [{lo}-{hi}]",
                      position=i, leave=True, ncols=80)

    ks = Ss(sd, sm, pbs)
    mw = 16
    LG.info("Workers=%d  TotalTarget=%d  MemLimit=%dGB", mw, ttg, ML)
    print(f"[MAIN] Workers={mw}")
    for t in TR: print(f"  {t}: {ks.ccn(t)}/{TR[t]['target']}")

    oc, fc, wc = 0, 0, 0
    try:
        with ThreadPoolExecutor(max_workers=mw, thread_name_prefix="W") as pl:
            fts = {}
            rm = sum(TR[t]['target'] - ks.ccn(t) for t in TR)
            for _ in range(min(mw, rm)):
                if not mok(): break
                tr = ks.pkt()
                if tr is None: break
                fts[pl.submit(wk, wc, ks, tr)] = wc; wc += 1

            while fts and not ks.azc():
                dn, _ = wait(fts.keys(), return_when=FIRST_COMPLETED, timeout=60)
                if not dn:
                    LG.warning("No tasks completed in 60s, pending=%d", len(fts)); continue
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
                        LG.info("STATUS ok=%d fail=%d │ low=%d/%d high=%d/%d │ RSS=%.1fGB",
                                oc, fc, ks.ccn("moderate_low"), TR["moderate_low"]["target"],
                                ks.ccn("moderate_high"), TR["moderate_high"]["target"], rg)
                while len(fts) < mw and not ks.azc():
                    if not mok(): LG.warning("Memory pressure, pausing spawn"); break
                    tr = ks.pkt()
                    if tr is None: break
                    fts[pl.submit(wk, wc, ks, tr)] = wc; wc += 1
            for f in fts: f.cancel()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted — shutting down..."); LG.warning("KeyboardInterrupt")
    except Exception as e:
        LG.error("MAIN crash: %s", e, exc_info=True)

    for pb in pbs.values(): pb.close()
    LG.info("═" * 70)
    LG.info("FINAL COMPONENT DISTRIBUTION")
    for tn in TR:
        tc = max(1, ks.tcp[tn])
        LG.info("── %s (%d samples) ──", tn, ks.ccn(tn))
        LG.info("  %-14s %8s %8s", "Component", "Actual%", "Count")
        for k in CK:
            LG.info("  %-14s %7.2f%% %7d", k, ks.cpc[tn][k] / tc * 100, ks.cpc[tn][k])
    LG.info("═" * 70)
    mg = (f"DONE: low={ks.ccn('moderate_low')}/{TR['moderate_low']['target']}, "
          f"high={ks.ccn('moderate_high')}/{TR['moderate_high']['target']}, "
          f"ok={oc}, fail={fc}, workers={wc}")
    LG.info(mg); print(f"\n[MAIN] {mg}")
