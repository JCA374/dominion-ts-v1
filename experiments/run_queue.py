"""Run the match arms of queue.txt one after another (each line: match.py arguments; '#' comments). An arm whose
results/match_<name>.json says done is skipped, and so is one whose match is running right now (another lane or an
arm started by hand); an unfinished one resumes. Re-reads the file after every arm, so arms can be added while it
runs. Lanes: --lanes 2 --lane 0/1 = two runners side by side, each taking the next free arm (--workers each).

    python experiments/run_queue.py --lane 0 --lanes 2 --workers 4     (systemd user units ts-lane0, ts-lane1)
"""
import argparse
import json, os, shlex, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


ap = argparse.ArgumentParser()
ap.add_argument("--lane", type=int, default=0)
ap.add_argument("--lanes", type=int, default=1)
ap.add_argument("--workers", type=int, default=7)
A = ap.parse_args()


def running(name):
    return subprocess.run(["pgrep", "-f", f"match.py .*--name {name} "], capture_output=True).returncode == 0


def pending():
    time.sleep(5 * A.lane)                       # lanes starting together take different arms
    for line in open(os.path.join(ROOT, "queue.txt")):
        line = line.split("#")[0].strip()
        if not line:
            continue
        args = shlex.split(line)
        name = args[args.index("--name") + 1]
        f = os.path.join(ROOT, "results", f"match_{name}.json")
        if os.path.exists(f) and json.load(open(f)).get("done") or running(name):
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
    if "--workers" in args:
        i = args.index("--workers")
        args = args[:i] + args[i + 2:]
    r = subprocess.call([PY, "-u", os.path.join(ROOT, "match.py"), *args, "--workers", str(A.workers),
                         *(["--resume"] if have else [])], cwd=ROOT)
    if r:
        print(time.strftime("%H:%M"), "arm", name, "failed", r, flush=True)
        break
