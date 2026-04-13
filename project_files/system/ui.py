"""Terminal UI helpers – no external dependencies."""

import os
import sys
import textwrap
from typing import Any, Callable, List, Optional, Tuple

# ── ANSI colours ──────────────────────────────────────────────────────────────
_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text

def bold(t: str)    -> str: return _c("1", t)
def dim(t: str)     -> str: return _c("2", t)
def green(t: str)   -> str: return _c("32", t)
def yellow(t: str)  -> str: return _c("33", t)
def red(t: str)     -> str: return _c("31", t)
def cyan(t: str)    -> str: return _c("36", t)
def magenta(t: str) -> str: return _c("35", t)
def blue(t: str)    -> str: return _c("34", t)

# ── Layout helpers ─────────────────────────────────────────────────────────────
def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80

def hr(char: str = "─") -> None:
    print(dim(char * _term_width()))

def banner(title: str) -> None:
    w = _term_width()
    pad = max(0, (w - len(title) - 4) // 2)
    print()
    hr("═")
    print(bold(cyan(f"{'═' * pad}  {title}  {'═' * pad}")))
    hr("═")
    print()

def section(title: str) -> None:
    print()
    print(bold(yellow(f"▸ {title}")))
    hr()

def success(msg: str) -> None:
    print(green(f"  ✔  {msg}"))

def warn(msg: str) -> None:
    print(yellow(f"  ⚠  {msg}"))

def error(msg: str) -> None:
    print(red(f"  ✘  {msg}"))

def info(msg: str) -> None:
    print(cyan(f"  ℹ  {msg}"))

# ── Menus ──────────────────────────────────────────────────────────────────────
def menu(
    title: str,
    items: List[Tuple[str, Any]],
    *,
    allow_back: bool = True,
    back_label: str = "← Back",
    zero_exit: bool = False,
    exit_label: str = "Exit",
) -> Optional[Any]:
    """
    Display a numbered menu and return the associated value.

    items: list of (label, value) pairs.
    Returns None when the user picks Back/Exit.
    """
    section(title)
    for i, (label, _) in enumerate(items, 1):
        print(f"  {bold(str(i))}.  {label}")
    back_n = len(items) + 1
    if allow_back:
        print(f"  {bold(str(back_n))}.  {dim(back_label)}")
    zero_n = 0
    if zero_exit:
        print(f"  {bold('0')}.  {dim(exit_label)}")
    print()

    while True:
        try:
            raw = input(bold("→ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if not raw:
            continue

        if zero_exit and raw == "0":
            return None

        try:
            n = int(raw)
        except ValueError:
            warn("Please enter a number.")
            continue

        if allow_back and n == back_n:
            return None
        if 1 <= n <= len(items):
            return items[n - 1][1]
        lo = "0" if zero_exit else "1"
        hi = str(back_n) if allow_back else str(len(items))
        warn(f"Invalid choice. Enter {lo}–{hi}.")


def prompt(question: str, default: str = "") -> str:
    """Simple text prompt with optional default."""
    hint = f" [{dim(default)}]" if default else ""
    try:
        val = input(f"  {bold(question)}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val if val else default


def confirm(question: str, default: bool = True) -> bool:
    """Yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {bold(question)} [{dim(hint)}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return default


def paginate(
    items: List[str],
    title: str = "",
    page_size: int = 20,
) -> None:
    """Display a long list with paging."""
    total = len(items)
    for i in range(0, total, page_size):
        chunk = items[i : i + page_size]
        if title and i == 0:
            section(title)
        for line in chunk:
            print(line)
        if i + page_size < total:
            remaining = total - i - page_size
            try:
                input(dim(f"  -- {remaining} more. Press Enter --"))
            except (EOFError, KeyboardInterrupt):
                print()
                return


def table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a simple aligned table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(str(cell)))

    sep = "  " + "  ".join("─" * w for w in widths)
    header_line = "  " + "  ".join(bold(h.ljust(widths[i])) for i, h in enumerate(headers))
    print(dim(sep))
    print(header_line)
    print(dim(sep))
    for row in rows:
        line = "  " + "  ".join(str(row[j]).ljust(widths[j]) for j in range(len(headers)))
        print(line)
    print(dim(sep))
