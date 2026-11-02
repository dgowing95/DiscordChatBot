"""Settings and their docs stay in step (names only, so it stays cheap).

- Every env var core reads has an AGENTS.md env-table row and an .env.example
  line (a value or a comment), and every AGENTS.md row names a variable that
  something reads (core, the other services, the voice sidecar, compose).
- Every values.yaml key is used by a chart template, so an unused value fails.
- No old name from settings_check.REMOVED is still set by the chart, compose
  or .env.example, where it would make core refuse to start.

Run from the repo root: pytest core/tests/settings_docs_tests.py
"""
import ast
import re
from pathlib import Path

from classes.settings_check import REMOVED

ROOT = Path(__file__).resolve().parents[2]
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*[A-Z0-9]$")

# Read by core but set by the platform, not by an operator.
NOT_SETTINGS = {
    "APP_VERSION",  # baked into the image by release.yaml (documented, but not in .env.example)
    "OPENAI_AGENTS_DISABLE_TRACING",  # the SDK's own; tracing is switched off in code
}
# Documented in AGENTS.md but read by something this test does not scan.
READ_ELSEWHERE = {
    "DOCKER_HOST",  # the docker library
}


def _env_name_args(func) -> int:
    """How many leading arguments of this call are env names: 1 for
    os.environ.get / os.getenv and the _env* helpers, 2 for env_or (primary
    and fallback), 0 for anything else."""
    if isinstance(func, ast.Attribute):
        if func.attr in ("get", "getenv") and (
                (isinstance(func.value, ast.Attribute) and func.value.attr == "environ")
                or (isinstance(func.value, ast.Name) and func.value.id == "os")):
            return 1
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return 0
    return 2 if name == "env_or" else 1 if name.startswith("_env") else 0


def env_names_read_by_core() -> set[str]:
    names = set()
    files = [ROOT / "core" / "main.py", *sorted((ROOT / "core" / "classes").glob("*.py"))]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _env_name_args(node.func):
                for arg in node.args[:_env_name_args(node.func)]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and ENV_NAME.match(arg.value):
                        names.add(arg.value)
            elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                  and node.value.attr == "environ" and isinstance(node.slice, ast.Constant)):
                names.add(node.slice.value)
    return names


def agents_env_rows() -> set[str]:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = text.split("### Environment variables", 1)[1].split("\n## ", 1)[0]
    names = set()
    for line in table.splitlines():
        if not line.startswith("| `"):
            continue
        first = line.split("|")[1]
        names.update(re.findall(r"`([A-Z][A-Z0-9_]+)`", first))
    return names


def env_example_names() -> set[str]:
    names = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#?\s*([A-Z][A-Z0-9_]+)=", line)
        if m:
            names.add(m.group(1))
    return names


def source_text() -> str:
    """Everything that can read a setting: services, sidecar, compose, e2e."""
    parts = []
    for pattern in ("core/main.py", "core/classes/*.py", "speechservice/*.py", "diffusionservice/*.py",
                    "browserservice/*.py", "voicesidecar/*.js", "voicesidecar/src/**/*.js",
                    "docker-compose.yaml"):
        for path in ROOT.glob(pattern):
            if "node_modules" not in path.parts:
                parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_every_env_var_core_reads_is_documented():
    read = env_names_read_by_core() - NOT_SETTINGS
    assert read, "the AST walk found nothing: the reader patterns are out of date"
    rows, example = agents_env_rows(), env_example_names()
    assert not read - rows, f"read by core but missing from AGENTS.md's env table: {sorted(read - rows)}"
    assert not read - example, f"read by core but missing from .env.example: {sorted(read - example)}"


def test_every_documented_env_var_is_read_somewhere():
    text = source_text()
    unused = sorted(name for name in agents_env_rows() - READ_ELSEWHERE
                    if not re.search(rf"(?<![A-Z0-9_]){name}(?![A-Z0-9_])", text))
    assert not unused, f"AGENTS.md documents settings nothing reads: {unused}"


def values_key_paths(text: str) -> list[str]:
    """Dotted paths of every key in a block-style values.yaml (no YAML library
    in the test deps; the chart's file is plain nested maps)."""
    paths, stack = [], []  # stack of (indent, key)
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or line.lstrip().startswith("-"):
            continue
        m = re.match(r"^(\s*)([A-Za-z_][\w-]*):(\s|$)", line)
        if not m:
            continue
        indent = len(m.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group(2)))
        paths.append(".".join(key for _, key in stack))
    return paths


def test_every_values_key_is_used_by_a_template():
    chart = ROOT / "charts" / "dis-ai-bot"
    templates = "\n".join(p.read_text(encoding="utf-8") for p in (chart / "templates").iterdir()
                          if p.name != "_removed.tpl")  # it names old keys on purpose
    used = set(re.findall(r"\.Values\.([A-Za-z_][\w.]*)", templates))
    # A template that hands over a whole map (toYaml, range, index) uses every key below it.
    used_prefixes = {u.rstrip(".") for u in used}
    unused = []
    for path in values_key_paths((chart / "values.yaml").read_text(encoding="utf-8")):
        parts = path.split(".")
        if not any(".".join(parts[:i]) in used_prefixes for i in range(1, len(parts) + 1)):
            if not any(u.startswith(path + ".") for u in used_prefixes):
                unused.append(path)
    assert not unused, f"values.yaml keys no template uses: {unused}"


def test_no_removed_name_is_still_set():
    chart = ROOT / "charts" / "dis-ai-bot" / "templates"
    configmap = (chart / "configmap.yaml").read_text(encoding="utf-8")
    secret = (chart / "secret.yaml").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
    example = env_example_names()
    found = []
    for name in REMOVED:
        if re.search(rf"^\s*{name}:", configmap + "\n" + secret, re.M):
            found.append(f"{name} (chart)")
        if re.search(rf"\$\{{{name}[:}}-]", compose):
            found.append(f"{name} (docker-compose.yaml)")
        if name in example:
            found.append(f"{name} (.env.example)")
    assert not found, f"old names still set, which would stop core from starting: {found}"
