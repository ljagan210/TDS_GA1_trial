"""Acme Appliances ledger Q&A service.

The question is turned into a small structured query (by an LLM if AIPIPE_TOKEN is set,
otherwise by keyword rules); all arithmetic is done here in Python on the real data.
"""
import calendar
import json
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

BASE = os.environ.get("LEDGER_BASE", (
    "https://exam.sanand.workers.dev/questionData?email=25f2007463%40ds.study.iitm.ac.in"
    "&quizSign=XDi%2FjWZUa7oF6RPgMCN8e%2FYU5MttgwPS6TazsrMopPieW3b1nXO5m6q13MT7mXlG2WHRENNwJkBOfyEradirHHmDxIj4xeJRXH0RfT8q7GTze9dH4BITIOclZpbDfxNweOW9C3wSIAVOE%2BZZ%2FoLWA3Jn1wX%2FtfuaqxyEBKvreJucbzU650NUnaRCf7lecGxkVnHwy3KMTmZvHxQoLlJnK%2BweyFKQgq1WNVUABdV0DjQyIFOcG9itgF9M1UaRs9cXpVoKq%2Bd%2BBGrJSM8F4N322nzzXyC06673kZMSxu6N4VgGsFzOamL%2BaOTOuPBnyBxQ6b7urv%2BLPbIaqzYKZd0Yeg%3D%3D"
    "&questionId=q-ledger-agent-server"
))
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "").strip()
AIPIPE_URL = os.environ.get("AIPIPE_URL", "https://aipipe.org/openai/v1/chat/completions")
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "gpt-4.1-mini")
IST = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata has no DST

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------------------------------------------------------ data
_cache = {"orders": None, "rates": None, "at": 0}
_lock = threading.Lock()


def _get(path):
    url = BASE + "&path=" + urllib.request.quote(path, safe="")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (ledger-agent)", "Accept": "application/json, */*"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode("utf-8")


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load():
    with _lock:
        if _cache["orders"] is not None and time.time() - _cache["at"] < 600:
            return _cache["orders"], _cache["rates"]
        rates = json.loads(_get("/rates"))["usd_per_unit"]
        latest = {}
        for line in _get("/export").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_updated"] = _ts(row["updated_at"])
            cur = latest.get(row["id"])
            if cur is None or row["_updated"] > cur["_updated"]:
                latest[row["id"]] = row
        orders = []
        for row in latest.values():
            created = _ts(row["created_at"]).astimezone(IST)
            row["_date"] = created.date()
            row["_usd"] = Decimal(str(row["amount"])) * Decimal(str(rates[row["currency"]]))
            orders.append(row)
        _cache.update(orders=orders, rates=rates, at=time.time())
        return orders, rates


# ------------------------------------------------------------------ query spec
METRICS = ["revenue", "refunds", "net_revenue", "order_count", "units", "customer_count",
           "avg_order_value", "refund_count", "refund_rate"]
GROUPS = ["product", "region", "customer", "month", "currency"]

SPEC_HELP = f"""Convert the user's question about an appliance order ledger into JSON with keys:
metric: one of {METRICS}
  revenue = USD value of paid orders; refunds = USD value of refunded orders;
  net_revenue = revenue minus refunds; order_count = number of orders;
  units = total quantity; customer_count = distinct customers;
  avg_order_value = revenue / number of paid orders; refund_count = number of refunded orders;
  refund_rate = refunded orders / (paid + refunded orders), as a fraction.
status: "paid", "refunded", "void" or null. Only set it for order_count/units/customer_count
  when the question restricts by status (e.g. "paid orders"); else null.
region: one of {{REGIONS}} or null.  product: one of {{PRODUCTS}} or null.
customer: customer id like "C1006" or null.  currency: original currency "USD"/"EUR"/"INR" or null
  (only when the question filters orders by their original currency).
start, end: inclusive dates "YYYY-MM-DD" for the period asked about (e.g. March 2026 ->
  2026-03-01..2026-03-31; Q2 2026 -> 2026-04-01..2026-06-30), or null if no period.
group_by: one of {GROUPS} or null - set when asking WHICH product/region/customer/month.
rank: "top" or "bottom" (with group_by); n: how many to return (default 1).
IMPORTANT: questions starting "How many"/"How much"/"What was the total" are never group_by questions,
even if they contain words like "at least" or "most". Use group_by only for "which/what X had the most...".
Examples:
"How many different customers placed at least one paid order for a Toaster?" ->
 {{"metric":"customer_count","status":"paid","product":"Toaster","group_by":null}}
"Which region brought in the least revenue in Q1 2026?" ->
 {{"metric":"revenue","group_by":"region","rank":"bottom","n":1,"start":"2026-01-01","end":"2026-03-31"}}
"How many Kettles were sold in April 2026?" -> {{"metric":"units","product":"Kettle","start":"2026-04-01","end":"2026-04-30"}}
Return only the JSON object."""


def llm_spec(question, regions, products):
    prompt = SPEC_HELP.replace("{REGIONS}", str(regions)).replace("{PRODUCTS}", str(products))
    body = json.dumps({
        "model": AIPIPE_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}],
    }).encode()
    req = urllib.request.Request(AIPIPE_URL, data=body, headers={
        "Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=9) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    return json.loads(content)


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def rule_spec(question, regions, products):
    q = question.lower()
    spec = {"metric": None, "status": None, "region": None, "product": None, "customer": None,
            "currency": None, "start": None, "end": None, "group_by": None, "rank": None, "n": 1}
    for r in regions:
        if re.search(rf"\b{re.escape(r.lower())}\b", q):
            spec["region"] = r
    for p in sorted(products, key=len, reverse=True):
        if p.lower() in q or p.lower().replace(" ", "") in q.replace(" ", ""):
            spec["product"] = p
            break
    m = re.search(r"\bc\d{3,}\b", q)
    if m:
        spec["customer"] = m.group(0).upper()
    for cur in ("EUR", "INR"):
        if re.search(rf"\b{cur.lower()}\b", q) or (cur == "EUR" and "euro" in q) or (cur == "INR" and "rupee" in q):
            spec["currency"] = cur
    year = int(re.search(r"\b(20\d\d)\b", q).group(1)) if re.search(r"\b(20\d\d)\b", q) else 2026
    qm = re.search(r"\bq([1-4])\b", q)
    if qm:
        k = int(qm.group(1))
        spec["start"], spec["end"] = f"{year}-{3*k-2:02d}-01", f"{year}-{3*k:02d}-{calendar.monthrange(year, 3*k)[1]}"
    else:
        found = []
        for w in re.findall(r"[a-z]+", q):
            if w in MONTHS and w != "may":
                found.append(MONTHS[w])
        if re.search(r"\bmay\b\s*(,?\s*20\d\d)|\b(in|of|during|for)\s+may\b", q):
            found.append(5)
        found = sorted(set(found))
        if found:
            a, b = found[0], found[-1]
            spec["start"] = f"{year}-{a:02d}-01"
            spec["end"] = f"{year}-{b:02d}-{calendar.monthrange(year, b)[1]}"
    qs = re.sub(r"\bat (least|most)\b", " ", q)
    is_amount_question = bool(re.match(r"\s*(how many|how much|what (is|was|were) the (total|number|count|sum|amount|average)|total|count|number of)", qs))
    if not is_amount_question and re.search(r"\bwhich\b|\bwhat (product|region|customer|month)\b|\btop\b|\bbest\b|\bmost\b|\bhighest\b|\blargest\b|\bbiggest\b|\blowest\b|\bleast\b|\bworst\b|\bsmallest\b|\bfewest\b", qs):
        for g in ("product", "region", "customer", "month"):
            if g in qs:
                spec["group_by"] = g
                break
        if spec["group_by"]:
            spec["rank"] = "bottom" if re.search(r"lowest|least|fewest|worst|smallest|bottom", qs) else "top"
    status = None
    for st, pat in (("paid", r"\bpaid\b"), ("refunded", r"\brefunded\b"), ("void", r"\bvoid(ed)?\b|\bcancel")):
        if re.search(pat, qs):
            status = st
            break
    if "refund rate" in qs or ("rate" in qs and "refund" in qs):
        spec["metric"] = "refund_rate"
    elif re.search(r"\bcustomers\b|\bbuyers\b|\bclients\b", qs) and re.search(r"how many|number of|count", qs):
        spec["metric"] = "customer_count"
        spec["status"] = status or ("paid" if re.search(r"\b(bought|buy|purchased|spent|paid)\b", qs) else None)
    elif "refund" in qs and re.search(r"how many|number of|count", qs) and not re.search(r"how much|amount|value|usd|\$", qs):
        spec["metric"] = "refund_count"
    elif "net" in qs.split() and re.search(r"revenue|sales", qs):
        spec["metric"] = "net_revenue"
    elif "refund" in qs:
        spec["metric"] = "refunds"
    elif re.search(r"average order|avg order|\baov\b|average .*(value|size|amount)", qs):
        spec["metric"] = "avg_order_value"
    elif re.search(r"\bunits\b|quantity|\bitems\b|how many .* (sold|bought|purchased)", qs) and not re.search(r"how many (\w+ )?orders", qs):
        spec["metric"] = "units"
        spec["status"] = status
    elif re.search(r"how many|number of|count of|order count", qs) and re.search(r"\border", qs):
        spec["metric"] = "order_count"
        spec["status"] = status
    else:
        spec["metric"] = "revenue"
    return spec


# ------------------------------------------------------------------ evaluation
def _money(d):
    return float(Decimal(d).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute(rows, metric, status=None):
    paid = [r for r in rows if r["status"] == "paid"]
    refunded = [r for r in rows if r["status"] == "refunded"]
    if metric == "revenue":
        return sum((r["_usd"] for r in paid), Decimal(0))
    if metric == "refunds":
        return sum((r["_usd"] for r in refunded), Decimal(0))
    if metric == "net_revenue":
        return compute(rows, "revenue") - compute(rows, "refunds")
    sel = [r for r in rows if status is None or r["status"] == status]
    if metric == "order_count":
        return len(sel)
    if metric == "refund_count":
        return len(refunded)
    if metric == "units":
        return sum(r["qty"] for r in (sel if status else paid))
    if metric == "customer_count":
        return len({r["customer"] for r in sel})
    if metric == "avg_order_value":
        return compute(rows, "revenue") / len(paid) if paid else Decimal(0)
    if metric == "refund_rate":
        d = len(paid) + len(refunded)
        return round(len(refunded) / d, 4) if d else 0
    raise ValueError(metric)


def fmt(metric, value):
    if metric in ("revenue", "refunds", "net_revenue", "avg_order_value"):
        return _money(value)
    return value


def answer(question):
    orders, _ = load()
    regions = sorted({o["region"] for o in orders})
    products = sorted({o["product"] for o in orders})
    spec = None
    if AIPIPE_TOKEN:
        try:
            spec = llm_spec(question, regions, products)
        except Exception:
            spec = None
    if not spec or spec.get("metric") not in METRICS:
        spec = rule_spec(question, regions, products)

    rows = orders
    if spec.get("region"):
        rows = [r for r in rows if r["region"].lower() == str(spec["region"]).lower()]
    if spec.get("product"):
        rows = [r for r in rows if r["product"].lower() == str(spec["product"]).lower()]
    if spec.get("customer"):
        rows = [r for r in rows if r["customer"].upper() == str(spec["customer"]).upper()]
    if spec.get("currency"):
        rows = [r for r in rows if r["currency"] == str(spec["currency"]).upper()]
    if spec.get("start"):
        s = datetime.fromisoformat(spec["start"]).date()
        rows = [r for r in rows if r["_date"] >= s]
    if spec.get("end"):
        e = datetime.fromisoformat(spec["end"]).date()
        rows = [r for r in rows if r["_date"] <= e]

    metric, status = spec["metric"], spec.get("status")
    group = spec.get("group_by")
    if group in GROUPS:
        buckets = defaultdict(list)
        for r in rows:
            key = r["_date"].strftime("%B %Y") if group == "month" else r[group]
            buckets[key].append(r)
        scored = [(compute(v, metric, status), k) for k, v in buckets.items()]
        reverse = spec.get("rank") != "bottom"
        scored.sort(key=lambda t: (-t[0], t[1]) if reverse else (t[0], t[1]))
        n = int(spec.get("n") or 1)
        names = [k for _, k in scored[:n]]
        return (names[0] if n == 1 else names), spec
    return fmt(metric, compute(rows, metric, status)), spec


@app.post("/{path:path}")
async def ask(request: Request, path: str = ""):
    body = await request.json()
    question = str(body.get("question", ""))
    try:
        value, spec = answer(question)
    except Exception as e:
        return {"answer": None, "error": f"{type(e).__name__}: {e}"}
    return {"answer": value}


@app.get("/{path:path}")
def health(path: str = ""):
    try:
        orders, rates = load()
        return {"status": "ok", "orders": len(orders), "llm": bool(AIPIPE_TOKEN)}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
