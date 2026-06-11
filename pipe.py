# pipe.py — DashScope LLM调用路由(流式/超时),
#            Worker流水线(合成→执行验证→多样性检测→共识投票→反向编译→保存)

import os, random, time, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI as _OAI
from core import SM, LG, TR, cc, fb, ps_sql, ps_nlq, rs, TP_CR, TP_SY, TP_RT

_dk = os.getenv("DASHSCOPE_API_KEY")
_cl = _OAI(api_key=_dk,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=600)

MSY = "qwen3.5-flash"
MVT = ["qwen3.6-plus", "glm-5.1", "kimi-k2.6", "deepseek-v4-flash", "qwen3.6-flash"]

_RST = {
    "qwen3.6-35b-a3b":   "Write a concise, formal English question",
    "glm-5.1":           "Write a detailed analytical question with clear reasoning intent",
    "kimi-k2.6":         "Write a simple, conversational English question",
    "qwen3.5-flash":     "Write a precise, technical question",
    "qwen3.5-27b":       "Write a business-oriented English question",
    "deepseek-v4-flash": "Write a clear English question covering all query conditions",
}
_ARM = list(_RST.keys())


def cl(pr, md):
    if not SM.acquire(timeout=180):
        LG.warning("LLM semaphore timeout for %s, skipping", md); return None
    LG.debug("LLM ← %s len=%d", md, len(pr))
    try:
        rp = _cl.chat.completions.create(
            model=md, messages=[{"role": "user", "content": pr}], stream=True)
        tx = ""; t0 = time.time(); la = t0
        for ck in rp:
            nw = time.time()
            if nw - t0 > 560 or nw - la > 120:
                LG.warning("LLM %s stream timeout (total=%.0fs idle=%.0fs)",
                           md, nw - t0, nw - la); break
            la = nw
            dt = ck.choices[0].delta if ck.choices else None
            if dt and dt.content: tx += dt.content
        if not tx: return None
        LG.debug("LLM → %s len=%d", md, len(tx)); return tx
    except Exception as e:
        LG.warning("LLM %s error: %s", md, e); return None
    finally: SM.release()


def _dg(sc, bd, lo, hi):
    ln = []
    if sc > hi:
        tp = sorted([(k, v) for k, v in bd.items() if v > 0],
                    key=lambda x: x[1], reverse=True)[:3]
        ln.append(f"OVER by {sc - hi}. Largest: " + ", ".join(f"{k}={v}" for k, v in tp))
        ln.append("REDUCE: remove excess AND/OR, simplify subqueries, cut functions.")
    elif sc < lo:
        ln.append(f"UNDER by {lo - sc}. Add more JOINs, functions, or a subquery/CTE.")
        ln.append("INCREASE: add a JOIN, use GROUP BY + aggregate, or add a predicate.")
    return "\n".join(ln) if ln else "Fine-tune component counts to hit target range."


def wk(wi, st, tt):
    if st.azc(): return False
    lo, hi = TR[tt]['range']
    di = st.gdi(tt)
    sc_ = st.smp[di]; dp = sc_['db_file_path']
    gs, fs, t3 = st.ggd(tt)
    uu = st.gut(tt, di)
    us = ", ".join(uu) if uu else "(all tables well-covered)"
    LG.info("W-%03d tier=%s db=%s top3=%s", wi, tt, di, t3)

    MR = 3; so, qo, sc, bd = None, None, -1, {}
    for at in range(1, MR + 1):
        if st.azc(): return False
        cr = TP_CR.format(lo=lo, hi=hi)
        if at == 1:
            pr = TP_SY.format(lo=lo, hi=hi, complexity_rules=cr,
                              schema_str=sc_['formatted_schema'],
                              comp_guide=gs, underused_tables=us, few_shot=fs)
        else:
            pr = TP_RT.format(attempt=at, max_attempts=MR, prev_score=sc,
                              lo=lo, hi=hi, breakdown=fb(sc, bd),
                              diagnosis=_dg(sc, bd, lo, hi), complexity_rules=cr,
                              schema_str=sc_['formatted_schema'],
                              comp_guide=gs, underused_tables=us)
        rw = cl(pr, MSY)
        if not rw: LG.info("W-%03d attempt %d: LLM returned None", wi, at); continue
        so = ps_sql(rw); qo = ps_nlq(rw)
        if not so: LG.info("W-%03d attempt %d: SQL parse fail", wi, at); continue
        if not qo: qo = f"Query about {di}"
        sc, bd = cc(so)
        LG.info("W-%03d attempt %d: score=%d target=[%d,%d]  %s",
                wi, at, sc, lo, hi, fb(sc, bd))
        if lo <= sc <= hi: break

    if not so or sc < 0:
        LG.info("W-%03d FAIL after %d attempts (score=%s)", wi, MR, sc); return False

    gsc, _ = cc(so); ft = st.cls(gsc)
    atr = None
    if tt in ft and not st.itc(tt): atr = tt
    else:
        for t in ft:
            if not st.itc(t): atr = t; break
    if atr is None:
        LG.info("W-%03d score=%d fits no incomplete tier", wi, gsc); return False
    if atr != tt:
        LG.info("W-%03d cross-tier: %s → %s (score=%d)", wi, tt, atr, gsc)

    rv = rs(dp, so)
    if rv is None:
        LG.info("W-%03d SQL exec failed → sef", wi)
        st.sef(atr, di, qo, so, sc, bd); return False
    LG.info("W-%03d SQL OK, %d rows", wi, len(rv))

    ok, em, sk = st.ckd(qo, so, sc_.get('schema_item_details', {}))
    if not ok:
        LG.info("W-%03d diversity reject → sdf", wi)
        st.sdf(atr, di, qo, so, sc, bd); return False

    nv = min(len(MVT), 3); vt = random.sample(MVT, nv)
    LG.info("W-%03d consensus voters=%s", wi, vt)
    cp = (f"Write SQLite SQL for the question. Output in [SQL-START]...[SQL-END].\n\n"
          f"Question: {qo}\nSchema:\n{sc_['formatted_schema']}")

    def _vt1(vm):
        try:
            vr = cl(cp, vm)
            if not vr: return False
            vs = ps_sql(vr)
            if not vs: return False
            vrs = rs(dp, vs, to=5)
            if vrs is None: return False
            return sorted(str(vrs)) == sorted(str(rv))
        except Exception: return False

    vrl = []
    try:
        with ThreadPoolExecutor(max_workers=nv, thread_name_prefix=f"V{wi}") as vp:
            vfs = {vp.submit(_vt1, m): m for m in vt}
            for fu in as_completed(vfs, timeout=120):
                try: vrl.append((vfs[fu], fu.result(timeout=5)))
                except Exception: vrl.append((vfs[fu], False))
    except Exception as e: LG.warning("W-%03d consensus pool error: %s", wi, e)

    np_ = sum(1 for _, p in vrl if p); nt = len(vrl)
    LG.info("W-%03d consensus: %d/%d passed", wi, np_, nt)
    if np_ <= nt / 2.0 or nt < 3:
        LG.info("W-%03d consensus FAIL → sef", wi)
        st.sef(atr, di, qo, so, sc, bd); return False

    def _rv1(m):
        sty = _RST[m]
        pr = (f"{sty} that would produce the following SQL. "
              f"Wrap it in [QUESTION-START]...[QUESTION-END].\n\nSQL: {so}")
        try:
            rr = cl(pr, m)
            if rr: return ps_nlq(rr)
        except Exception: pass
        return None

    cnq = []
    for bt in (_ARM[:4], _ARM[4:]):
        with ThreadPoolExecutor(max_workers=len(bt), thread_name_prefix=f"R{wi}") as rp:
            rfs = {rp.submit(_rv1, m): m for m in bt}
            for fu in as_completed(rfs, timeout=120):
                try:
                    q = fu.result(timeout=5)
                    if q and len(q) > 10: cnq.append(q)
                except Exception: pass

    if cnq:
        if len(cnq) == 1: qo = cnq[0]
        else:
            try:
                re_ = st.emm.encode(cnq, convert_to_tensor=True, batch_size=32)
                sm = torch.nn.functional.cosine_similarity(
                    re_.unsqueeze(0), re_.unsqueeze(1), dim=2)
                sm.fill_diagonal_(0)
                av = sm.sum(dim=1) / (len(cnq) - 1)
                bi = av.argmax().item(); qo = cnq[bi]
                LG.info("W-%03d reverse: %d candidates, picked #%d (avg_sim=%.3f)",
                        wi, len(cnq), bi, av[bi].item())
            except Exception as e:
                LG.warning("W-%03d reverse similarity error: %s", wi, e); qo = cnq[0]

    ok = st.svr(atr, di, qo, so, sc, bd, em, sk)
    LG.info("W-%03d %s tier=%s total=%d", wi,
            "SAVED" if ok else "skip", atr, st.ccn(atr))
    return ok
