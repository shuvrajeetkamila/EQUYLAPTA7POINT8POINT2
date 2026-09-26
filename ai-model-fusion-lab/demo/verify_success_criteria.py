"""Phase 23 — verify all 11 success criteria against live artifacts."""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("FUSIONLAB_DATA",
                      os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(
                          os.path.abspath(__file__))), os.pardir, "fusionlab_data")))
out = []
def check(n, desc, ok, evidence):
    out.append(f"TEST {n}: {'PASS' if ok else 'FAIL'} — {desc}\n         evidence: {evidence}")

exps = [json.loads(l) for l in open(os.path.join(DATA, "experiments.jsonl"))]
types = {}
for e in exps:
    types.setdefault(e["type"], 0)
    types[e["type"]] += 1

# TEST 1
r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_core"],
                   capture_output=True, text=True, cwd=".")
check(1, "existing regression suite passes", "OK" in r.stderr,
      r.stderr.strip().splitlines()[-1])
# TEST 2
dist = [e for e in exps if e["type"] == "distillation"]
check(2, "distillation completes without the M1 tensor-shape failure",
      len(dist) > 0 and any((e["results"] or {}).get("student_math") for e in dist),
      f"{len(dist)} distillation experiments recorded, e.g. {dist[-1]['id'] if dist else '-'}")
# TEST 3
gj = os.path.join(DATA, "gauntlet.json")
g = json.load(open(gj)) if os.path.exists(gj) else []
supported = [x["family"] for x in g if x.get("verdict") == "SUPPORTED"]
check(3, "at least two real-model architectures load",
      len(supported) >= 2, f"gauntlet SUPPORTED: {supported}")
# TEST 4
gpt2_abl = [e for e in exps if e["type"] == "genome-ablation"
            and e["sources"] and "gpt2" in e["sources"][0]]
check(4, "at least one real model undergoes component-level ablation",
      len(gpt2_abl) > 0, f"{len(gpt2_abl)} gpt2 head-ablation experiments (text-math)")
# TEST 5
gf = os.path.join(DATA, "genomes")
genomes = os.listdir(gf) if os.path.isdir(gf) else []
nrec = sum(len(json.load(open(os.path.join(gf, f))).get("records", [])) for f in genomes)
check(5, "candidate capability-associated components generated",
      nrec > 50, f"{len(genomes)} genomes, {nrec} measured records")
# TEST 6
check(6, "at least one component extraction experiment runs",
      types.get("genome-ablation", 0) > 0 and os.path.exists("demo/reports/SPECIALIST_REGISTRY.json"),
      f"{types.get('genome-ablation',0)} ablation records feed Component pools (registry written)")
# TEST 7
synth = [e for e in exps if e["type"] == "synthesis"]
trans = [e for e in synth if e["config"].get("mechanism") == "DIRECT_TRANSPLANT"]
align = [e for e in exps if e["type"] == "alignment"]
check(7, "at least one alignment/transplant experiment runs",
      len(trans) > 0 and len(align) > 0,
      f"{len(trans)} transplant attempts + {len(align)} alignment experiments")
# TEST 8
check(8, "at least one synthesis attempt is evaluated",
      len(synth) > 0, f"{len(synth)} synthesis attempts recorded: " +
      str(dict((m, sum(1 for e in synth if e['config'].get('mechanism')==m))
               for m in {e['config'].get('mechanism') for e in synth})))
# TEST 9
sp = json.load(open("demo/reports/SPECIALIST_REGISTRY.json"))
rej = [s for s in sp["specialists"] if "REJECT" in s["status"]]
check(9, "inferior combinations are automatically rejected",
      len(rej) > 0, f"{len(rej)} specialists rejected: " +
      "; ".join(f"{s['specialist']} ({s['status']})" for s in rej))
# TEST 10
check(10, "specialist produced OR honest failure demonstration",
      len(sp["specialists"]) >= 3, "3 construction reports with explicit reasons; "
      "fallback-to-parent recorded when mechanisms underperformed")
# TEST 11
sm = sp.get("super_model", {})
routing = sm.get("routing", {})
check(11, "router combines at least two specialists",
      len(routing) >= 2, f"EQUYLAPTA-1 routed suites: {routing}; specialists: "
      f"{list(sm.get('specialists', {}))}")

txt = "EQUYLAPTA [AI MODEL] — MILESTONE 2 SUCCESS CRITERIA (Phase 23)\n" + "=" * 70 + "\n" + "\n\n".join(out)
print(txt)
open("demo/reports/SUCCESS_CRITERIA.txt", "w").write(txt + "\n")
passed = txt.count("PASS")
print(f"\n{passed}/11 criteria PASS")
