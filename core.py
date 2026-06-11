# core.py — 环境初始化, 日志配置, 全局常量/Tier定义, sqlglot AST复杂度评分,
#            SQL与NLQ解析, 骨架/表名提取, SQL沙箱执行, LLM Prompt模板

import os, re, time, logging, sqlite3, threading, sys
import psutil
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv

os.environ["TRANSFORMERS_OFFLINE"] = "1"
load_dotenv()

RT = os.path.dirname(os.path.abspath(__file__))
DR = os.path.join(RT, "augmented_data")
os.makedirs(os.path.join(RT, "logs"), exist_ok=True)
os.makedirs(DR, exist_ok=True)

_ts = time.strftime('%m%d_%H%M')
LP = os.path.join(RT, "logs", f"aug_v8_spider_{_ts}.log")

LG = logging.getLogger("AugV8Spider")
LG.setLevel(logging.DEBUG)
LG.propagate = False
if LG.hasHandlers():
    LG.handlers.clear()
_fmt = logging.Formatter(
    '%(asctime)s │ %(levelname)-5s │ %(threadName)-14s │ %(message)s',
    datefmt='%H:%M:%S')
_fh = logging.FileHandler(LP, encoding='utf-8')
_fh.setLevel(logging.DEBUG); _fh.setFormatter(_fmt)
LG.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
LG.addHandler(_ch)
for _n in ("httpx", "openai", "anthropic"):
    logging.getLogger(_n).setLevel(logging.WARNING)
print(f"[STARTUP] Log → {LP}")
LG.info("═" * 70)
LG.info("V8 Spider Augmentation Pipeline ── Session Start")
LG.info("═" * 70)

SM = threading.Semaphore(25)
ML = 50
_mb = ML * 1024 ** 3

def mok():
    try: return psutil.Process().memory_info().rss < _mb
    except Exception: return True

CK = [
    'A_JOIN', 'B_Function', 'B_Logic', 'B_Arithmetic', 'B_CASE', 'B_Predicate',
    'C_GROUP', 'C_HAVING', 'D_Subquery', 'E_ORDER', 'E_LIMIT', 'E_TOP',
    'F_Window', 'G_SetOp', 'H_DISTINCT',
]
RK = [k for k in CK if k != 'E_TOP']

CG = {
    'A_JOIN':       "add JOIN clauses (INNER/LEFT/CROSS JOIN between multiple tables)",
    'B_Function':   "use functions (COUNT, AVG, SUM, SUBSTR, COALESCE, CAST, ROUND and other SQLite functions)",
    'B_Logic':      "add WHERE conditions with AND / OR / NOT operators",
    'B_Arithmetic': "include arithmetic expressions (+, -, *, /, %)",
    'B_CASE':       "use a CASE WHEN ... THEN ... ELSE ... END expression",
    'B_Predicate':  "use IN(...), BETWEEN, LIKE, EXISTS, or IS NULL predicates",
    'C_GROUP':      "include GROUP BY for aggregation",
    'C_HAVING':     "add HAVING clause after GROUP BY to filter groups",
    'D_Subquery':   "use a subquery in FROM/WHERE or a WITH ... AS (CTE)",
    'E_ORDER':      "use ORDER BY in the query",
    'E_LIMIT':      "use LIMIT in the query",
    'F_Window':     "use window function: ROW_NUMBER() / RANK() / LAG() OVER(...)",
    'G_SetOp':      "use UNION / INTERSECT / EXCEPT to combine queries",
    'H_DISTINCT':   "use SELECT DISTINCT to eliminate duplicates",
}

TR = {
    "moderate_low":  {"range": (3, 4), "seed_max": 2, "target": 500,
                      "output": "aug_v8_spider_moderate_3_4.json"},
    "moderate_high": {"range": (5, 7), "seed_max": 2, "target": 180,
                      "output": "aug_v8_spider_moderate_5_7.json"},
}

# ── AST复杂度 ──

class _Vx:
    def __init__(self, dp=0):
        self.dp = dp
        self.c = {k: 0 for k in CK}

    def vi(self, nd):
        if not isinstance(nd, exp.Expression): return
        if isinstance(nd, (exp.Subquery, exp.CTE)):
            self.c['D_Subquery'] += 1
            ix = nd.this
            if isinstance(ix, exp.Expression):
                ch = _Vx(self.dp + 1); ch.vi(ix); self._mg(ch)
            return
        self._sc(nd)
        for v in nd.args.values():
            if isinstance(v, list):
                for it in v: self.vi(it)
            else: self.vi(v)

    @property
    def tot(self): return sum(self.c.values())

    def _sc(self, n):
        if isinstance(n, exp.Join):        self.c['A_JOIN'] += 1
        if isinstance(n, exp.Func):        self.c['B_Function'] += 1
        if isinstance(n, (exp.And, exp.Or, exp.Not)):                   self.c['B_Logic'] += 1
        if isinstance(n, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)):self.c['B_Arithmetic'] += 1
        if isinstance(n, exp.Case):        self.c['B_CASE'] += 1
        if isinstance(n, (exp.In, exp.Between, exp.Like, exp.Exists, exp.Is)):
            self.c['B_Predicate'] += 1
            if isinstance(n, exp.Like) and n.args.get('escape') is not None:
                self.c['B_Predicate'] += 1
        if isinstance(n, exp.Select):
            if n.args.get('group'):    self.c['C_GROUP'] += 1
            if n.args.get('having'):   self.c['C_HAVING'] += 1
            if n.args.get('order'):    self.c['E_ORDER'] += 1
            if n.args.get('limit'):    self.c['E_LIMIT'] += 1
            if n.args.get('distinct'): self.c['H_DISTINCT'] += 1
        if isinstance(n, exp.Window):      self.c['F_Window'] += 1
        if isinstance(n, (exp.Union, exp.Intersect, exp.Except)): self.c['G_SetOp'] += 1

    def _mg(self, ch):
        for k in CK: self.c[k] += ch.c[k]


def cc(sq, dl='sqlite'):
    try:
        tr = sqlglot.parse_one(sq, dialect=dl)
        v = _Vx(); v.vi(tr)
        return v.tot, dict(v.c)
    except Exception:
        return -1, {k: 0 for k in CK}

def fb(sc, bd):
    return "  ".join([f"Total={sc}"] + [f"{k}={bd[k]}" for k in CK if bd.get(k, 0) > 0])

# ── 解析 ──

def ps_sql(tx):
    if not tx: return None
    for pt in [r"\[SQL-START\](.*?)\[SQL-END\]", r"```sql\s*\n(.*?)\n```",
               r"```\s*(?:sql)?\s*\n(.*?)\n```", r"(SELECT\s+.*?;?)(?:\n\n|$)"]:
        m = re.search(pt, str(tx), re.DOTALL | re.I)
        if m:
            r = re.sub(r'```\s*', '', m.group(1).strip())
            if "SELECT" in r.upper(): return r
    t = str(tx).strip()
    if t.upper().startswith("SELECT") and "FROM" in t.upper(): return t
    return None

def ps_nlq(tx):
    if not tx: return None
    for pt in [r"\[QUESTION-START\](.*?)\[QUESTION-END\]",
               r"(?:Question|问题)[:：]\s*(.*?)(?=\n\n|\[SQL|$)"]:
        m = re.search(pt, str(tx), re.DOTALL | re.I)
        if m:
            r = m.group(1).strip()
            if len(r) > 10: return r[:500]
    for ln in str(tx).split('\n'):
        ln = ln.strip()
        if ln and len(ln) > 15 and not ln.upper().startswith("SELECT"): return ln[:500]
    return None

def ex_sk(sq, sd):
    if not sq: return ""
    u = sq.upper()
    u = re.sub(r"'(.*?)'", "'[V]'", u)
    u = re.sub(r"\b\d+(\.\d+)?\b", "[N]", u)
    for t in sd.get('table_names_original', []):
        if t and t.upper() not in {"SELECT", "FROM", "WHERE", "JOIN", "ON", "AND"}:
            u = re.sub(rf"\b{re.escape(t.upper())}\b", "[T]", u)
    return re.sub(r'\s+', ' ', u).strip()

def ex_tb(sq):
    tb = set()
    for m in re.finditer(r'\bFROM\s+(\w+)', sq, re.I): tb.add(m.group(1).lower())
    for m in re.finditer(r'\bJOIN\s+(\w+)', sq, re.I): tb.add(m.group(1).lower())
    return tb

# ── SQL执行 ──

def rs(dp, sq, to=5):
    rv, er = [None], [None]
    def _f():
        try:
            cn = sqlite3.connect(dp, timeout=to)
            cn.execute("PRAGMA busy_timeout = 5000;")
            cn.execute("PRAGMA query_only = ON;")
            rv[0] = cn.execute(sq).fetchall(); cn.close()
        except Exception as e: er[0] = e
    t = threading.Thread(target=_f, daemon=True)
    t.start(); t.join(timeout=to)
    if t.is_alive() or er[0]: return None
    return rv[0]

# ── Prompt模板 ──

TP_CR = """\
### Complexity Scoring (sqlglot AST-based, each occurrence = +1)
| Component    | What counts                                          |
|--------------|------------------------------------------------------|
| A_JOIN       | Each JOIN clause                                     |
| B_Function   | Each function call (COUNT, AVG, SUBSTR, CAST...)     |
| B_Logic      | Each AND, OR, NOT                                    |
| B_Arithmetic | Each +, -, *, /, %                                   |
| B_CASE       | Each CASE expression                                 |
| B_Predicate  | Each IN, BETWEEN, LIKE, EXISTS, IS                   |
| C_GROUP      | GROUP BY present → +1                                |
| C_HAVING     | HAVING present → +1                                  |
| D_Subquery   | Each subquery/CTE (+1 self, inner components recurse)|
| E_ORDER      | ORDER BY present → +1                               |
| E_LIMIT      | LIMIT present → +1                                   |
| F_Window     | Each OVER() window clause                            |
| G_SetOp      | Each UNION / INTERSECT / EXCEPT                      |
| H_DISTINCT   | Each SELECT DISTINCT                                 |

TARGET: total complexity score must be {lo}–{hi}."""

TP_SY = """\
You are a Senior SQLite query architect.
Generate EXACTLY one SQL query with complexity {lo}–{hi}.

{complexity_rules}

### Database Schema
{schema_str}

### Component Emphasis (randomly selected for diversity)
The following components are selected for emphasis in this query.
STEP 1: Evaluate whether each component below is naturally applicable
        to this database schema (consider table relationships, column types,
        and meaningful query semantics).
STEP 2: For each FEASIBLE component, incorporate it into your SQL.
        Skip any that would create unnatural or meaningless queries.

{comp_guide}

### Schema Exploration
These tables/columns have been UNDEREXPLORED in existing queries.
Build your query around them to discover overlooked data patterns:
  Underused tables: {underused_tables}
Examine the schema above — focus on columns, relationships, and data
that the reference examples below did NOT touch.

### Reference Examples
{few_shot}

### Output Format (strictly follow — NO [THOUGHT] tag needed)
[QUESTION-START] A natural-language question the SQL answers [QUESTION-END]
[SQL-START] Your SQLite SQL here [SQL-END]
"""

TP_RT = """\
### RETRY — Attempt {attempt}/{max_attempts}
Previous SQL complexity = {prev_score} (target: {lo}–{hi}).
Breakdown: {breakdown}

Diagnosis:
{diagnosis}

{complexity_rules}

### Database Schema
{schema_str}

### Component Emphasis (randomly selected for diversity)
{comp_guide}

### Schema Exploration (underused tables)
  {underused_tables}

Adjust your SQL to land in [{lo}, {hi}]. Output:
[QUESTION-START] question [QUESTION-END]
[SQL-START] SQL [SQL-END]
"""
