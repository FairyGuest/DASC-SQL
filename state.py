# state.py — Spider Schema加载与格式化, JSON读写,
#             SharedState共享状态管理(多样性检测, Tier管理, 嵌入缓存, 数据持久化)

import os, random, json, math, threading
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from modelscope import snapshot_download
from core import CK, RK, CG, TR, DR, LG, cc, ex_sk, ex_tb


def lj(fp):
    try:
        if not os.path.exists(fp): LG.warning(f"File '{fp}' not found."); return []
        with open(fp, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        LG.error(f"Error loading JSON from '{fp}': {e}"); return []


def _fmt_sch(si):
    fs = "Table information:\n"
    tn = si.get('table_names_original', [])
    cn = si.get('column_names_original', [])
    ch = si.get('column_names', [])
    ct = si.get('column_types', [])
    pk = set(si.get('primary_keys', []))
    fk = si.get('foreign_keys', [])
    cbt = defaultdict(list)
    for i, (ti, nm) in enumerate(cn):
        if ti == -1: continue
        hr = ch[i][1] if i < len(ch) else nm
        tp = ct[i] if i < len(ct) else "TEXT"
        cbt[ti].append((nm, hr, tp, i in pk))
    for ti, nm in enumerate(tn):
        fs += f"Table name: {nm}\n"
        for c, h, tp, ip in cbt.get(ti, []):
            pt = " [PRIMARY KEY]" if ip else ""
            fs += f"  Column: {c} ({tp}){pt}"
            if h.lower() != c.lower(): fs += f" -- {h}"
            fs += "\n"
        fs += "\n"
    if fk:
        fs += "Foreign Keys:\n"
        for fc, rc in fk:
            if fc < len(cn) and rc < len(cn):
                fti, fcn = cn[fc]; rti, rcn = cn[rc]
                ft = tn[fti] if 0 <= fti < len(tn) else "?"
                rt = tn[rti] if 0 <= rti < len(tn) else "?"
                fs += f"  {ft}.{fcn} -> {rt}.{rcn}\n"
    return fs.strip()


def csm(tjp, drp):
    mp = {}
    al = lj(tjp)
    if not al: LG.error(f"No schemas loaded from {tjp}"); return {}
    for si in al:
        di = si.get('db_id', '')
        if not di: continue
        dfp = os.path.join(drp, di, f"{di}.sqlite")
        if not os.path.exists(dfp):
            at = os.path.join(drp, f"{di}.sqlite")
            if os.path.exists(at): dfp = at
            else: LG.warning(f"DB not found for '{di}' at '{dfp}', skipping"); continue
        mp[di] = {'formatted_schema': _fmt_sch(si), 'db_file_path': dfp,
                   'schema_item_details': si}
    return mp


class Ss:
    def __init__(self, asds, smp, pbs):
        self.lk = threading.RLock()
        self.smp = smp
        self.pbs = pbs
        self._dids = list(smp.keys())
        self.dt = {t: [] for t in TR}
        self.cev = {t: threading.Event() for t in TR}
        self.dbc = {t: defaultdict(int) for t in TR}
        self.qta = {t: max(1, math.ceil(TR[t]['target'] / max(1, len(smp)))) for t in TR}
        self.efd = {t: [] for t in TR}
        self.dfd = {t: [] for t in TR}
        self.ssk = set()
        self.gem = None
        self._mc = 20000
        self.cpc = {t: {k: 0 for k in CK} for t in TR}
        self.tcp = {t: 0 for t in TR}
        self.tbu = {t: defaultdict(lambda: defaultdict(int)) for t in TR}

        LG.info("Loading embedding model...")
        print("[STARTUP] Loading embedding model...")
        mp = snapshot_download('AI-ModelScope/bge-large-en-v1.5', revision='master')
        self.emm = SentenceTransformer(mp, device='cpu')
        LG.info("Embedding model ready.")
        print("[STARTUP] Embedding model loaded.")

        LG.info("Computing seed complexities...")
        print("[STARTUP] Computing seed complexities...")
        self.sds = []
        nc = 0
        for s in asds:
            sq = s.get('query', '')
            if not sq: continue
            sc, bd = cc(sq); nc += 1
            if sc < 0: continue
            s['_comp'] = sc; s['_bd'] = bd
            if sc <= 2: self.sds.append(s)
        LG.info("Seeds computed: %d total │ eligible=%d", nc, len(self.sds))
        print(f"[STARTUP] Seeds: total={nc}, eligible={len(self.sds)}")

        for tn, tc in TR.items():
            op = os.path.join(DR, tc['output'])
            if os.path.exists(op):
                try:
                    ht = lj(op); lo, hi = tc['range']
                    for it in ht:
                        sq = it.get('SQL'); di = it.get('db_id')
                        if sq and di in smp:
                            gs, _ = cc(sq)
                            if not (lo <= gs <= hi): continue
                            self.dt[tn].append(it)
                            self.dbc[tn][di] += 1
                            sd = smp[di].get('schema_item_details', {})
                            self.ssk.add(ex_sk(sq, sd))
                            bd = it.get('breakdown', {})
                            for k in CK: self.cpc[tn][k] += bd.get(k, 0)
                            self.tcp[tn] += sum(bd.get(k, 0) for k in CK)
                            for tb in ex_tb(sq): self.tbu[tn][di][tb] += 1
                    self.pbs[tn].update(len(self.dt[tn]))
                    LG.info("Loaded %d historical items for %s", len(self.dt[tn]), tn)
                except Exception as e:
                    LG.warning("History load failed for %s: %s", tn, e)

        _nq = []
        for s in asds:
            q = s.get('question', '')
            if q and len(q) > 5: _nq.append(q)
        for tn in TR:
            for it in self.dt[tn]:
                q = it.get('question', '')
                if q and len(q) > 5: _nq.append(q)
        if _nq:
            LG.info("Batch-encoding %d NLQs for diversity...", len(_nq))
            print(f"[STARTUP] Encoding {len(_nq)} NLQs for diversity...")
            self.gem = self.emm.encode(_nq, convert_to_tensor=True,
                                       batch_size=128, show_progress_bar=True)
            LG.info("NLQ embeddings ready: %d vectors", self.gem.shape[0])
            print(f"[STARTUP] NLQ embeddings: {self.gem.shape[0]} vectors")
        for t in TR: LG.info("%s: %d/%d", t, len(self.dt[t]), TR[t]['target'])

    def _aem(self, em):
        if self.gem is None: self.gem = em.unsqueeze(0)
        else:
            self.gem = torch.cat([self.gem, em.unsqueeze(0)])
            if self.gem.shape[0] > self._mc: self.gem = self.gem[-self._mc:]

    def itc(self, tr):
        with self.lk: return len(self.dt[tr]) >= TR[tr]['target']

    def azc(self):
        with self.lk: return all(len(self.dt[t]) >= TR[t]['target'] for t in TR)

    def ccn(self, tr):
        with self.lk: return len(self.dt[tr])

    def pkt(self):
        with self.lk:
            cd = [(t, TR[t]['target'] - len(self.dt[t])) for t in TR
                  if TR[t]['target'] - len(self.dt[t]) > 0]
            if not cd: return None
            tl = sum(r for _, r in cd)
            rv = random.random() * tl; cm = 0
            for t, rm in cd:
                cm += rm
                if rv <= cm: return t
            return cd[-1][0]

    def cls(self, sc):
        return [t for t, cf in TR.items() if cf['range'][0] <= sc <= cf['range'][1]]

    def ggd(self, tr):
        t3 = random.sample(RK, 3)
        gs = "\n".join(f"{i}. **{k}**: {CG.get(k, k)}" for i, k in enumerate(t3, 1))
        with self.lk:
            pl = self.sds
            rv = [s for s in pl if any(s.get('_bd', {}).get(k, 0) > 0 for k in t3)]
            if len(rv) < 2: rv = pl
            sp = random.sample(rv, min(2, len(rv))) if rv else []
            fs = "\n\n".join(f"Q: {s.get('question', 'N/A')}\nSQL: {s.get('query', '')}"
                             for s in sp) if sp else "(No examples available)"
        return gs, fs, t3

    def gut(self, tr, di):
        with self.lk:
            sd = self.smp[di].get('schema_item_details', {})
            at = [t for t in sd.get('table_names_original', []) if t]
            ug = self.tbu[tr][di]
            return sorted(at, key=lambda t: ug.get(t.lower(), 0))[:3]

    def gdi(self, tr):
        with self.lk:
            av = [d for d in self._dids if self.dbc[tr][d] < self.qta[tr]]
            return random.choice(av) if av else random.choice(self._dids)

    def ckd(self, qu, sq, sd):
        em = self.emm.encode(qu, convert_to_tensor=True)
        with self.lk:
            if self.gem is not None and self.gem.shape[0] > 0:
                sm = torch.max(util.cos_sim(em, self.gem)).item()
                if sm >= 0.9: return False, None, None
            sk = ex_sk(sq, sd)
            if sk in self.ssk: return False, None, None
        return True, em, sk

    def sef(self, tr, di, qu, sq, sc, bd):
        with self.lk:
            self.efd[tr].append({"db_id": di, "question": qu, "SQL": sq,
                                 "complexity": sc, "breakdown": bd})
            lo, hi = TR[tr]['range']
            op = os.path.join(DR, f"aug_v8_spider_{lo}_{hi}_exec_fail.json")
            tp = op + ".tmp"
            try:
                with open(tp, 'w', encoding='utf-8') as f:
                    json.dump(self.efd[tr], f, indent=2, ensure_ascii=False)
                os.replace(tp, op)
            except Exception as e: LG.error("sef failed: %s", e)

    def sdf(self, tr, di, qu, sq, sc, bd):
        with self.lk:
            self.dfd[tr].append({"db_id": di, "question": qu, "SQL": sq,
                                 "complexity": sc, "breakdown": bd})
            lo, hi = TR[tr]['range']
            op = os.path.join(DR, f"aug_v8_spider_{lo}_{hi}_diversity_fail.json")
            tp = op + ".tmp"
            try:
                with open(tp, 'w', encoding='utf-8') as f:
                    json.dump(self.dfd[tr], f, indent=2, ensure_ascii=False)
                os.replace(tp, op)
            except Exception as e: LG.error("sdf failed: %s", e)

    def svr(self, tr, di, qu, sq, sc, bd, em, sk):
        with self.lk:
            if len(self.dt[tr]) >= TR[tr]['target']: return False
            self.dt[tr].append({"db_id": di, "question": qu, "SQL": sq,
                                "complexity": sc, "breakdown": bd})
            self.ssk.add(sk); self.dbc[tr][di] += 1
            for k in CK: self.cpc[tr][k] += bd.get(k, 0)
            self.tcp[tr] += sum(bd.get(k, 0) for k in CK)
            for tb in ex_tb(sq): self.tbu[tr][di][tb] += 1
            self._aem(em)
            op = os.path.join(DR, TR[tr]['output']); tp = op + ".tmp"
            try:
                with open(tp, 'w', encoding='utf-8') as f:
                    json.dump(self.dt[tr], f, indent=2, ensure_ascii=False)
                os.replace(tp, op)
            except Exception as e: LG.error("Save failed: %s", e); return False
            cn = len(self.dt[tr]); self.pbs[tr].update(1)
            LG.info("SAVED [%s/%s] comp=%d  %d/%d", tr, di, sc, cn, TR[tr]['target'])
            if cn >= TR[tr]['target']: self.cev[tr].set()
            return True
