"""Run the match arms of queue.txt one after another (each line: match.py arguments; '#' comments). An arm whose
results/match_<name>.json says done is skipped; an unfinished one resumes. Re-reads the file after every arm, so arms
can be added while it runs.

    python experiments/queue.py          (as the systemd user unit ts-queue)
"""
import json, os, shlex, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def pending():
    for line in open(os.path.join(ROOT, "queue.txt")):
        line = line.split("#")[0].strip()
        if not line:
            continue
        args = shlex.split(line)
        name = args[args.index("--name") + 1]
        f = os.path.join(ROOT, "results", f"match_{name}.json")
        if os.path.exists(f) and json.load(open(f)).get("done"):
            continue
        return name, args, os.path.exists(f.replace(".json", ".jsonl"))
    return None


while True:
    p = pending()
    if p is None:
        print(time.strftime("%H:%M"), "queue empty", flush=True)
        break
    name, args, have = p
    print(time.strftime("%H:%M"), "arm", name, flush=True)
    r = subprocess.call([PY, "-u", os.path.join(ROOT, "match.py"), *args, *(["--resume"] if have else [])], cwd=ROOT)
    if r:
        print(time.strftime("%H:%M"), "arm", name, "failed", r, flush=True)
        break
