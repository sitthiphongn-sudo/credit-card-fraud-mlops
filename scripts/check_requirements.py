"""Fail CI when a direct dependency is not pinned to one exact version."""

import re
from pathlib import Path

PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?==[A-Za-z0-9][A-Za-z0-9.!+_-]*$")


def check(path: Path) -> list[str]:
    errors = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line and not PIN.fullmatch(line):
            errors.append(f"{path}:{line_number}: dependency needs an exact == pin: {line}")
    return errors


if __name__ == "__main__":
    problems = check(Path("requirements.txt"))
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)
    print("All direct dependencies are pinned")
