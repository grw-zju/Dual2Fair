import os
import re
import sys


FORBIDDEN = {
    'fusion_alpha': re.compile(r'fusion_alpha'),
    'max_disadv_users': re.compile(r'max_disadv_users'),
    'max_ot_items': re.compile(r'max_ot_items'),
    'strict_bilevel_class': re.compile(r'class\s+BiLevelOptimizer'),
    'disabled_nystrom': re.compile(r'Nystrom.*disabled|Nyström.*disabled', re.I),
}

WHITELIST = {'models/dual2fair/bilevel_opt.py', 'ALIGNMENT_REPORT.md',
             'scripts/audit_revision_alignment.py'}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    violations = []
    for directory, _, files in os.walk(root):
        relative_directory = os.path.relpath(directory, root)
        if relative_directory.startswith(('.git', 'baseline', 'tests')):
            continue
        for name in files:
            if not name.endswith(('.py', '.yaml', '.md')):
                continue
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, root)
            if relative in WHITELIST:
                continue
            with open(path, 'r', errors='ignore') as handle:
                for number, line in enumerate(handle, 1):
                    for key, pattern in FORBIDDEN.items():
                        if pattern.search(line):
                            violations.append((relative, number, key, line.strip()))
    for violation in violations:
        print(f'{violation[0]}:{violation[1]} [{violation[2]}] {violation[3]}')
    return int(bool(violations))


if __name__ == '__main__':
    sys.exit(main())
