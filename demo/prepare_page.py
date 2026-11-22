from pathlib import Path
root=Path(__file__).resolve().parent
(root/"dist/index.html").write_bytes((root/"page_template.html").read_bytes())
