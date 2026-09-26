"""[WORKING] Web GUI server — stdlib only (no framework deps).

Single-file dashboard with inline CSS/JS (works in sandboxed previews),
JSON API for the lab state, and background pipeline runs with live logs.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE = {
    "pipeline": {"running": False, "stage": "idle", "log_tail": [], "progress": 0.0},
    "transfer": {"running": False, "log_tail": [], "last": None},
    "last_super_eval": None,
}
_DASHBOARD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def DATA_DIR():
    return os.environ.get("FUSIONLAB_DATA",
                          os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                              os.path.dirname(os.path.abspath(__file__)))),
                          "fusionlab_data")))


def genome_action_worker(action: str, quick: bool):
    """Run the corresponding M2 stage on demand (micro scale)."""
    STATE["pipeline"] = {"running": True, "stage": f"M2:{action}", "log_tail": [],
                         "progress": 0.05}
    buf = STATE["pipeline"]["log_tail"]

    def log(msg=""):
        buf.append(str(msg))
        del buf[:-400]
        print(msg, flush=True)

    try:
        import numpy as np
        from models.synthetic import (build_profile_corpus, load_model,
                                      master_tokenizer, save_model, train_micro)
        from experiments.tracker import ExperimentTracker
        from capability_genome.discovery import discover
        from capability_genome.genome import CapabilityGenome
        from component_extraction.transplant import extract_pool
        from component_synthesis.engine import (build_specialist_v2, render_report,
                                                attempt_alignment_gate)
        from adapters.lowrank import fit_projection_lowrank
        from adapters.projection import collect_hidden
        import numpy as np

        DATA = DATA_DIR()
        tracker = ExperimentTracker(DATA)
        tok = master_tokenizer()
        names = ["math-wiz", "code-smith", "logic-owl", "polyglot", "generalist", "weak-base"]
        models = {n: load_model(n) for n in names}
        os.makedirs(os.path.join(DATA, "genomes"), exist_ok=True)
        caps = {"math": "micro-math", "coding": "micro-code", "reasoning": "micro-reason",
                "language": "micro-lang", "multilingual": "micro-multi"}

        if action in ("discover", "pipeline"):
            for n in ["math-wiz", "code-smith", "logic-owl", "generalist", "weak-base"]:
                log(f"discovering genome: {n}")
                g = discover(models[n], caps, tracker, max_items=25, log=None)
                g.save(os.path.join(DATA, "genomes", f"{n}.json"))
                log("  " + g.summary().splitlines()[0] +
                    f" | candidates: {len(g.candidates(min_abs_delta=4))}")
        if action in ("extract", "pipeline"):
            genomes = {n: CapabilityGenome.load(os.path.join(DATA, "genomes", f"{n}.json"))
                       for n in ["math-wiz", "code-smith", "logic-owl", "generalist", "weak-base"]
                       if os.path.exists(os.path.join(DATA, "genomes", f"{n}.json"))}
            for cap, suite in [("math", "micro-math"), ("coding", "micro-code")]:
                pool = extract_pool(genomes, models, capability=cap, suite=suite, top_k=4)
                log(f"pool[{cap}]: " + (", ".join(f"{c.source_model}:{c.region}" for c in pool) or "empty"))
        if action in ("align", "pipeline"):
            probes = build_profile_corpus("generalist", 80, seed=77)
            gate = attempt_alignment_gate(models["math-wiz"], models["weak-base"], probes)
            log(f"alignment gate math-wiz->weak-base: {gate['after_ridge']} "
                f"(sufficient={gate['alignment_sufficient']})")
        if action in ("synthesize", "test", "pipeline"):
            genomes = {n: CapabilityGenome.load(os.path.join(DATA, "genomes", f"{n}.json"))
                       for n in ["math-wiz", "code-smith", "logic-owl", "generalist", "weak-base"]
                       if os.path.exists(os.path.join(DATA, "genomes", f"{n}.json"))}
            pool = extract_pool(genomes, models, capability="math", suite="micro-math", top_k=4)
            rep = build_specialist_v2("EQUYLAPTA-math", "math", "micro-math",
                                      ["micro-code", "micro-lang"], models, pool,
                                      tracker, max_items=25, log=log)
            log(render_report(rep))
        STATE["pipeline"]["stage"] = f"M2:{action} complete"
        STATE["pipeline"]["progress"] = 1.0
    except Exception as e:
        import traceback
        buf.append("ERROR: " + repr(e))
        buf.append(traceback.format_exc()[-800:])
        STATE["pipeline"]["stage"] = f"failed: {e}"
    finally:
        STATE["pipeline"]["running"] = False


def pipeline_worker(quick: bool, real: bool):
    STATE["pipeline"] = {"running": True, "stage": "starting", "log_tail": [], "progress": 0.0}
    buf = STATE["pipeline"]["log_tail"]

    def log(msg=""):
        buf.append(str(msg))
        del buf[:-400]
        print(msg, flush=True)

    try:
        _run_pipeline(quick, real, log)
        STATE["pipeline"]["stage"] = "complete"
        STATE["pipeline"]["progress"] = 1.0
    except Exception as e:
        import traceback
        buf.append("ERROR: " + repr(e))
        buf.append(traceback.format_exc()[-800:])
        STATE["pipeline"]["stage"] = f"failed: {e}"
    finally:
        STATE["pipeline"]["running"] = False


def transfer_worker(source: str, target: str, capability: str, quick: bool):
    """M3 Transfer Lab (spec §22, §29): runs the CLI transfer experiment and
    streams its log; the last rendered report + verdict land in STATE."""
    STATE["transfer"] = {"running": True, "log_tail": [], "last": None}
    buf = STATE["transfer"]["log_tail"]

    def log(msg=""):
        buf.append(str(msg))
        del buf[:-400]

    try:
        from experiments.transfer_runner import run as run_exp
        R = run_exp(source, target, capability, "auto", n_items=20 if quick else 40,
                    quick=quick)
        STATE["transfer"]["last"] = {
            "source": source, "target": target, "capability": capability,
            "verdict": R.get("verdict", {}),
            "arms": {k: v for k, v in R.get("arms", {}).items() if v},
            "component": R.get("component", {}),
        }
        log("DONE — verdict: " + json.dumps(R.get("verdict", {}), default=str))
    except Exception as e:
        import traceback
        log("ERROR: " + repr(e))
        log(traceback.format_exc()[-800:])
        STATE["transfer"]["last"] = {"error": str(e)}
    finally:
        STATE["transfer"]["running"] = False


def _components_payload():
    """Data-connected component map for the Transfer Lab (§30)."""
    import glob
    from component_transfer.component import index_library
    models = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR(), "genomes", "*.json"))):
        try:
            d = json.load(open(f))
            models.append({"name": d.get("model_name"),
                           "n_records": d.get("n_records", 0),
                           "baselines": d.get("baselines", {})})
        except Exception:
            continue
    lin_path = os.path.join(DATA_DIR(), "lineage.json")
    lineage = json.load(open(lin_path)) if os.path.exists(lin_path) else {}
    lib = {}
    try:
        lib = index_library()
    except Exception:
        pass
    reports = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "demo", "reports")
    matrix = None
    headline = None
    for name, key in (("transfer_matrix.json", "matrix"),
                      ("TRANSFER_HEADLINE.txt", "headline")):
        p = os.path.join(reports, name)
        if os.path.exists(p):
            if key == "matrix":
                try:
                    matrix = json.load(open(p))
                except Exception:
                    pass
            else:
                try:
                    headline = open(p).read()
                except Exception:
                    pass
    return {"genomes": models, "lineage": lineage,
            "library": {k: [c.get("component_id") for c in v]
                        for k, v in lib.items()},
            "library_count": sum(len(v) for v in lib.values()),
            "matrix": matrix, "headline": headline}


def _run_pipeline(quick, real, log):
    import numpy as np
    from models.synthetic import (build_profile_corpus, data_dir, master_tokenizer,
                                  save_model, train_micro)
    from models.registry import Registry, compatibility_matrix
    from models.autopsy import autopsy, tree_map
    from benchmarking.suites import all_micro_suites, freeze_private
    from benchmarking.harness import eval_model, capability_matrix
    from ablation.ablator import localize_capabilities, contribution_report
    from experiments.tracker import ExperimentTracker
    from licensing.registry import check_configuration

    t0 = time.time()
    suites = all_micro_suites(n=40)
    suite_names = [s.name for s in suites]
    freeze_private(suites)
    tracker = ExperimentTracker(data_dir())
    reg = Registry(os.path.join(data_dir(), "registry.json"))
    tok = master_tokenizer()
    n_stages = 6.0

    log("STAGE 1/6 training source micro-models ...")
    STATE["pipeline"]["stage"] = "training sources"
    STATE["pipeline"]["progress"] = 1 / n_stages
    profiles = ["math-wiz", "code-smith", "logic-owl", "polyglot", "generalist", "weak-base"]
    models = {}
    for p in profiles:
        m = train_micro(p, p, epochs=40 if quick else 150, batch=128, lr=1e-2,
                        d_model=64, n_layers=2, n_heads=4, log=log)
        save_model(m); reg.register(m, "Apache-2.0"); models[p] = m
        log(f"  trained {p} val={m.train_history[-1]['val_loss']:.2f}")

    log("STAGE 2/6 autopsy ...")
    STATE["pipeline"]["stage"] = "autopsy"; STATE["pipeline"]["progress"] = 2 / n_stages
    log(tree_map(autopsy(models["math-wiz"])))

    log("STAGE 3/6 compatibility ...")
    STATE["pipeline"]["stage"] = "compatibility"; STATE["pipeline"]["progress"] = 3 / n_stages
    for cm in compatibility_matrix(reg):
        log(f"  {cm['pair']}: shape_ok={cm['shape_compatible']} merge={cm['merge_compatibility']}")
        break
    log("  ... (full matrix in registry)")

    log("STAGE 4/6 capability benchmarking ...")
    STATE["pipeline"]["stage"] = "benchmarking"; STATE["pipeline"]["progress"] = 4 / n_stages
    results = {}
    for name, m in models.items():
        results[name] = eval_model(m, suite_names, max_items=40)
        log(f"  {name} done")
    log(capability_matrix(results))

    log("STAGE 5/6 ablation + localization ...")
    STATE["pipeline"]["stage"] = "ablation"; STATE["pipeline"]["progress"] = 5 / n_stages
    local = localize_capabilities(models["generalist"], ["micro-math", "micro-code"],
                                  max_items=40, log=log)
    log(contribution_report(local))

    log("STAGE 6/6 merges + specialists ...")
    STATE["pipeline"]["stage"] = "merging"; STATE["pipeline"]["progress"] = 0.92
    from merging.engine import build_specialist
    out = build_specialist(models["generalist"], [models["math-wiz"], models["code-smith"]],
                           "micro-math", tracker, max_items=40, log=log)
    log(f"  math specialist verdict: {out['verdict']}")
    lic = check_configuration([{"name": "sources", "license": "Apache-2.0"}])
    log(f"  license: {lic['license_compatibility']}")
    log(f"DONE in {time.time()-t0:.0f}s")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = open(_DASHBOARD, "rb").read()
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path.startswith("/api/genome"):
            import glob
            files = sorted(glob.glob(os.path.join(DATA_DIR(), "genomes", "*.json")))
            if not files:
                return self._send(200, {"genome": None,
                                        "note": "no genomes yet — run the M2 pipeline"})
            recs = []
            for f in files:
                try:
                    d = json.load(open(f))
                    recs.append(d)
                except Exception:
                    continue
            biggest = max(recs, key=lambda d: d.get("n_records", 0))
            return self._send(200, {"genome": biggest,
                                    "available": [r.get("model_name") for r in recs],
                                    "note": f"{len(recs)} genomes on disk; showing "
                                    f"{biggest.get('model_name')} (most records)"})
        if self.path.startswith("/api/pattern_genome"):
            try:
                pg_path = os.path.join(DATA_DIR(), "pattern_genome.json")
                if not os.path.exists(pg_path):
                    return self._send(200, {"note": "no pattern genome yet — "
                                             "run demo/run_milestone4.py"})
                from pattern_genome.registry import PatternRegistry
                reg = PatternRegistry.load(pg_path)
                circs = list(reg.circuits.values()) \
                    if isinstance(reg.circuits, dict) else list(reg.circuits)
                return self._send(200, {
                    "summary": reg.summary_lines(),
                    "n_patterns": len(reg.results),
                    "circuits": [{k: c.get(k) for k in
                                  ("circuit_id", "kind", "model", "capability",
                                   "evidence_level", "status")}
                                 for c in circs][:40]})
            except Exception as e:
                return self._send(500, {"error": str(e)})
        if self.path.startswith("/api/components"):
            try:
                return self._send(200, _components_payload())
            except Exception as e:
                return self._send(500, {"error": str(e)})
        if self.path.startswith("/api/transfer/status"):
            return self._send(200, STATE["transfer"])
        if self.path.startswith("/api/state"):
            from models.synthetic import data_dir, saved_names
            from experiments.tracker import ExperimentTracker
            from models.registry import Registry
            try:
                tr = ExperimentTracker(data_dir())
                reg = Registry(os.path.join(data_dir(), "registry.json"))
                lb = sorted(tracker := tr.all(), key=lambda r: -sum(
                    v for v in r["results"].values() if isinstance(v, (int, float))) or 0)[:12]
                return self._send(200, {
                    "pipeline": STATE["pipeline"],
                    "models": reg.models,
                    "experiments": [tr.all()[-i - 1] for i in range(min(10, len(tr.all())))],
                })
            except Exception as e:
                return self._send(200, {"pipeline": STATE["pipeline"], "models": {},
                                        "experiments": [], "error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/api/transfer"):
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            source = body.get("source") or "A-math"
            target = body.get("target") or "nomath-base"
            capability = body.get("capability") or "math"
            quick = bool(body.get("quick", True))
            if STATE["transfer"]["running"]:
                return self._send(409, {"error": "a transfer run is already active"})
            threading.Thread(target=transfer_worker,
                             args=(source, target, capability, quick),
                             daemon=True).start()
            return self._send(200, {"started": True, "source": source,
                                    "target": target, "capability": capability,
                                    "quick": quick})
        if self.path.startswith("/api/genome/"):
            action = self.path.rsplit("/", 1)[-1]
            valid = {"discover", "extract", "align", "synthesize", "test", "pipeline"}
            if action not in valid:
                return self._send(404, {"error": f"unknown genome action {action}"})
            if STATE["pipeline"]["running"]:
                return self._send(409, {"error": "pipeline already running"})
            quick = action != "pipeline"
            threading.Thread(target=genome_action_worker, args=(action, quick),
                             daemon=True).start()
            return self._send(200, {"started": True, "action": action,
                                    "message": f"M2 '{action}' launched — progress streams in the pipeline panel"})
        if self.path.startswith("/api/run"):
            if STATE["pipeline"]["running"]:
                return self._send(409, {"error": "pipeline already running"})
            quick = "quick" in self.path
            threading.Thread(target=pipeline_worker, args=(quick, True), daemon=True).start()
            return self._send(200, {"started": True})
        if self.path.startswith("/api/route"):
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            q = body.get("query", "")
            try:
                from router.router import LearnedRouter, router_training_data
                r = LearnedRouter()
                X, Y = router_training_data()
                r.fit(X, Y)
                caps, conf = r.route(q)
                return self._send(200, {"query": q, "activate": caps, "confidence": conf})
            except Exception as e:
                return self._send(500, {"error": str(e)})
        return self._send(404, {"error": "not found"})


def serve(host: str = "0.0.0.0", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Fusion Lab GUI -> http://{host}:{port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
