"""Add a note to the site log (Log tab), or a decision (Log tab + Overview "Decisions"):
    python experiments/journal.py "title" "text"
    python experiments/journal.py --decision "title" "what was chosen and why" ["what was rejected"]
"""
import json, os, sys, time
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'journal.jsonl')
args = sys.argv[1:]
entry = {'t': time.strftime('%Y-%m-%d %H:%M:%S')}
if args and args[0] == '--decision':
    entry.update(kind='decision', title=args[1], text=args[2], rejected=args[3] if len(args) > 3 else '')
else:
    entry.update(kind='journal', title=args[0], text=args[1])
with open(path, 'a') as f:
    f.write(json.dumps(entry) + '\n')
print('noted')
