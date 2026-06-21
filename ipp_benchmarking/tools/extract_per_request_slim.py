#!/usr/bin/env python3
"""Tolerantly extract a slim per-request record list from inference-perf's giant
per_request_lifecycle_metrics.json (OCP runs).

inference-perf stores the full streamed `response` per record, so the file is
100MB–1GB and is usually truncated mid-write (json.load -> "Unterminated string").
We raw_decode object-by-object and stop at the first incomplete one, so every
COMPLETE record before the tail is recovered. Per record we keep only:
  t   = start_time
  lat = end_time - start_time
  fail= bool(error)
  m   = served model: "small" (Qwen3-8B) / "big" (Qwen3-32B) / null (fails: empty response)

Output: <arm>/per_request_slim.json  (a few hundred KB; re-plottable instantly).

    extract_per_request_slim.py <big_per_request.json> <out_slim.json>
"""
import json, sys

def extract(path):
    dec = json.JSONDecoder()
    txt = (sys.stdin.read() if path == "-" else open(path, "r", errors="ignore").read())
    i = txt.find("{")
    out = []
    while i != -1:
        try:
            rec, j = dec.raw_decode(txt, i)
        except json.JSONDecodeError:
            break  # truncated tail
        st, et = rec.get("start_time"), rec.get("end_time")
        if isinstance(st, (int, float)) and isinstance(et, (int, float)):
            resp = rec.get("response") or ""
            m = "big" if "Qwen3-32B" in resp else ("small" if "Qwen3-8B" in resp else None)
            out.append({"t": st, "lat": et - st, "fail": bool(rec.get("error")), "m": m})
        k = txt.find("{", j)
        if k == -1:
            break
        i = k
    out.sort(key=lambda r: r["t"])
    return out


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        # raw_decode must recover complete records and stop cleanly at a truncated tail
        import tempfile, os
        good = '[{"start_time":1.0,"end_time":2.0,"error":null,"response":"x Qwen3-8B y"},' \
               '{"start_time":3.0,"end_time":9.0,"error":"boom","response":""},' \
               '{"start_time":2.0,"end_time":4.0,"error":null,"response":"Qwen3-32B"},{"start_time":5.0,"end'  # truncated
        p = tempfile.mktemp(suffix=".json"); open(p, "w").write(good)
        r = extract(p); os.unlink(p)
        assert len(r) == 3, r  # 3 complete records; the 4th (truncated) is dropped
        # sorted by t (1,2,3): small, big(32B), then the failed one (empty response -> m None)
        assert [x["m"] for x in r] == ["small", "big", None], r
        assert r[2]["fail"] is True and r[2]["lat"] == 6.0, r
        print("selfcheck ok:", r)
        sys.exit(0)
    rows = extract(sys.argv[1])
    json.dump(rows, open(sys.argv[2], "w"))
    n = len(rows); s = sum(x["m"] == "small" for x in rows); b = sum(x["m"] == "big" for x in rows)
    print(f"{sys.argv[2]}: {n} records  8B={s} 32B={b} fail={sum(x['fail'] for x in rows)}")
