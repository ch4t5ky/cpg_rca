"""Compare two coverage snapshots; exit 1 if visible decreases or lost increases.

Usage: python3 -m eval.compare OLD.json NEW.json

Totals and every service of OLD are checked, on union and pooled counts. A
missing service or key raises. A smaller FLOW lowers reachable and lost
together, so read the reachable column too.
"""

import json
import sys
from pathlib import Path

SHOWN = ("reachable", "visible", "lost", "pooled_reachable", "pooled_visible", "pooled_lost")


def main(argv):
    old, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv)
    rows = [("total", old["totals"], new["totals"])]
    rows += [(name, s, new["services"][name]) for name, s in old["services"].items()]
    regressed = False
    print("| service | " + " | ".join(SHOWN) + " | status |\n|" + "---|" * (len(SHOWN) + 2))
    for name, o, n in rows:
        bad = any(n[k] < o[k] for k in ("visible", "pooled_visible")) or any(
            n[k] > o[k] for k in ("lost", "pooled_lost"))
        regressed |= bad
        print(f"| {name} | " + " | ".join(f"{o[k]} -> {n[k]}" for k in SHOWN)
              + f" | {'REGRESSION' if bad else 'ok'} |")
    return int(regressed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
