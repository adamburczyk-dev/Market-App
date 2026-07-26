#!/usr/bin/env python3
"""Reject non-ASCII characters in PowerShell scripts.

Windows PowerShell 5.1 reads a .ps1 file without a BOM using the system ANSI
code page, so a UTF-8 em dash arrives as mojibake ("â€”") and derails the
parser — the reported error points at a perfectly valid line further down.
Keeping these scripts pure ASCII sidesteps encoding entirely, in every
PowerShell version and on every code page.
"""

import sys


def check(path: str) -> list[str]:
    problems = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            for col, char in enumerate(line, 1):
                if ord(char) > 127:
                    problems.append(
                        f"{path}:{lineno}:{col} non-ASCII {char!r} (U+{ord(char):04X}) "
                        f"— PowerShell 5.1 will misread this file"
                    )
    return problems


def main(paths: list[str]) -> int:
    problems = [problem for path in paths for problem in check(path)]
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
