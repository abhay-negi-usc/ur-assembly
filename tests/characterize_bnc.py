"""Characterization harness for the bnc_assembly port: pin the DRY-RUN trace, diff every step.

Usage:
    python tests/characterize_bnc.py            # diff against the golden; exit 1 on drift
    python tests/characterize_bnc.py --update   # rewrite the golden (state WHY in the commit)

What it pins. The full app is run with `--dry-run --yes --set assembly.target_source=kinematic`
and `q` piped to stdin: the black dry-run frame means the scanner finds nothing and prompts, and
`q` takes the deterministic abort path. The normalized output — construction order, config
resolution, speed-scale phases, gripper/arm step labels, gate texts, the scanner hand-off — is
the golden. It deliberately covers app CONSTRUCTION and the entry sequencing end-to-end; the
deep phases (engage, clocking) are pinned by their own unit tests, which migrate with each
extraction step (see docs/bnc_port_log.md).

Normalization: timestamps, experiment-folder names, and absolute paths vary per run and are
scrubbed; everything else must match byte-for-byte.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, 'tests', 'goldens', 'bnc_dryrun_trace.txt')
CMD = [sys.executable, '-m', 'urlab.apps.bnc_assembly', '--dry-run', '--yes',
       '--set', 'run.target_source=kinematic']


def normalize(text):
    out = []
    for ln in text.splitlines():
        ln = re.sub(r'^\[\d\d:\d\d:\d\d\] ', '', ln)                      # log timestamps
        ln = re.sub(r'bnc_assembly_\d{8}_\d{6}', 'bnc_assembly_<STAMP>', ln)
        ln = re.sub(r'[A-Za-z]:\\[^\s]*ur-assembly', '<REPO>', ln)        # absolute paths
        out.append(ln.rstrip())
    return '\n'.join(out).rstrip() + '\n'


def capture():
    proc = subprocess.run(CMD, cwd=ROOT, input='q\n', capture_output=True, text=True,
                          timeout=600)
    return normalize(proc.stdout + proc.stderr), proc.returncode


def main():
    trace, code = capture()
    if '--update' in sys.argv:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, 'w', newline='\n') as fh:
            fh.write(trace)
        print(f'golden updated ({len(trace.splitlines())} lines, app exit {code})')
        return 0
    if not os.path.isfile(GOLDEN):
        print('no golden yet -- run with --update first')
        return 2
    golden = open(GOLDEN, newline='\n').read()
    if trace == golden:
        print(f'trace matches the golden ({len(trace.splitlines())} lines, app exit {code})')
        return 0
    import difflib
    sys.stdout.writelines(difflib.unified_diff(
        golden.splitlines(keepends=True), trace.splitlines(keepends=True),
        'golden', 'current'))
    print('\nTRACE DRIFTED. If intended, re-run with --update and explain in the commit.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
