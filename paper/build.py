"""Build the submitted PDF (main.pdf, Apart Research template) from report.md.

Restructures the Markdown into the Apart template's section order, converts it with pandoc
(--natbib), and compiles main.tex with latexmk. Run from paper/:  python3 build.py
"""
import re, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "report.md").read_text()

def sections(text):
    parts = re.split(r"(?m)^(# .*)$", text)
    out = {}
    for i in range(1, len(parts), 2):
        out[parts[i][2:].strip()] = parts[i + 1].strip("\n")
    return out

body = SRC[SRC.index("# 1. Introduction"):]
S = sections(body)
def get(prefix):
    return next(v for k, v in S.items() if k.startswith(prefix))

def strip_sub(md):  # "## 4.1 Title" -> "## Title"; "## C.1 Title" -> "## Title"
    return re.sub(r"(?m)^## (?:\d+\.\d+|[A-Z]\.\d+) ", "## ", md)

def remap(md):
    # Source appendix letters follow report.md; 'Appendix @X' is passed through as 'Appendix X'.
    md = re.sub(r"\(Section 2\)", "(Section 3.1)", md)
    md = re.sub(r"\(Section 3\)", "(Section 3.2)", md)
    md = re.sub(r"environments of Section 2\b", "environments of Section 3.1", md)
    for old, new in (("A", "@B"), ("E", "@C"), ("B", "@D"), ("C", "@E"), ("D", "@F")):
        md = re.sub(rf"Appendix {old}\b", f"Appendix {new}", md)
    return md.replace("Appendix @", "Appendix ")

disc = get("6. Discussion")
disc = disc.replace("**Limitations.**", "## Limitations\n\n").replace("**Future work.**", "## Future Work\n\n")

main_md = "\n\n".join([
    "# Introduction\n\n" + get("1. Introduction"),
    "# Related Work\n\n" + get("5. Related Work"),
    "# Methods\n\n## Problem Formulation\n\n" + get("2. Problem Formulation"),
    "## Testbeds and Data\n\n" + get("3. Experimental Setup"),
    "# Results\n\n" + strip_sub(get("4. Results")),
    "# Discussion and Limitations\n\n" + disc,
    "# Conclusion\n\n" + get("7. Conclusion"),
    "# Code and Data {-}\n\n" + (HERE / "code_and_data.md").read_text(),
])
app_md = "\n\n".join([
    "# Limitations and Dual-Use / Ethical Considerations\n\n" + (HERE / "dual_use.md").read_text(),
    "# Selection and Grouping Rules\n\n" + get("Appendix A"),
    "# Testbed Details\n\n" + get("Appendix E"),
    "@@NEWPAGE@@\n\n# The GTG-1002 Campaign\n\n" + get("Appendix B"),
    "@@NEWPAGE@@\n\n# Examples from the Dataset\n\n" + strip_sub(get("Appendix C")),
    "# Models and Monitor Prompts\n\n" + get("Appendix D"),
    "# LLM Usage Statement\n\n" + get("AI Use Disclosure"),
])
abstract = re.search(r"(?ms)^abstract: \|\n(.*?)^---", SRC).group(1)
abstract = "\n".join(l.strip() for l in abstract.splitlines())
(HERE / "abstract.md").write_text(abstract)
subprocess.run(["pandoc", "abstract.md", "-f", "markdown", "-t", "latex", "-o", "abstract.tex"], cwd=HERE, check=True)
for name, md in (("body", main_md), ("appendix", app_md)):
    md = remap(md).replace("\\clearpage", "").replace("@@NEWPAGE@@", "\\clearpage")
    (HERE / f"{name}.md").write_text(md)
    subprocess.run(["pandoc", f"{name}.md", "-f", "markdown", "-t", "latex", "--natbib",
                    "--top-level-division=section", "-o", f"{name}.tex"], cwd=HERE, check=True)

# System prompts in the models appendix: "\emph{label}" + quote -> titled prompt box.
tex = (HERE / "appendix.tex").read_text()
i = tex.index("Models and Monitor Prompts")
j = tex.index("\\section", i)
part = re.sub(r"\\emph\{(.*?)\}(.*?)\n\n\\begin\{quote\}\n(.*?)\\end\{quote\}",
              lambda m: "\\begin{promptbox}{" + (m.group(1).rstrip(".") + m.group(2)).strip() + "}\n" + m.group(3) + "\\end{promptbox}",
              tex[i:j], flags=re.S)
(HERE / "appendix.tex").write_text(tex[:i] + part + tex[j:])
subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=HERE, check=True)
print("built", HERE / "main.pdf")
