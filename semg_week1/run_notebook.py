"""Execute every notebook cell in the project venv, then export a portable HTML view."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
for name, subdir in [('JUPYTER_CONFIG_DIR', 'jupyter_config'), ('JUPYTER_DATA_DIR', 'jupyter_data'), ('JUPYTER_RUNTIME_DIR', 'jupyter_runtime'), ('IPYTHONDIR', 'ipython'), ('MPLCONFIGDIR', 'matplotlib')]:
    target = ROOT / 'tmp' / subdir
    target.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(target)
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager

nb_path = ROOT / 'week1_semg.ipynb'
nb = nbformat.read(nb_path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, timeout=180, km=km, resources={'metadata': {'path': str(ROOT)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb, nb_path)
body, _ = HTMLExporter().from_notebook_node(nb)
(ROOT / 'output' / 'week1_report.html').write_text(body, encoding='utf-8')
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
print('PASS: all notebook cells executed; embedded figures and HTML exported.')
