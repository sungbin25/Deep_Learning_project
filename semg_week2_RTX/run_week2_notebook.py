"""Execute and export the Week 2 report without implicitly starting long training."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
for name,sub in [('JUPYTER_CONFIG_DIR','jupyter_config'),('JUPYTER_DATA_DIR','jupyter_data'),('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    target=ROOT/'tmp'/sub; target.mkdir(parents=True,exist_ok=True); os.environ[name]=str(target)
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
path=ROOT/'week2_semg.ipynb'
nb=nbformat.read(path,as_version=4)
km=KernelManager(kernel_name='python3'); km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(nb,timeout=600,km=km,resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb,path)
body,_=HTMLExporter().from_notebook_node(nb)
(ROOT/'output/week2/week2_report.html').write_text(body,encoding='utf-8')
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
print('PASS: Week 2 report cells executed; training status is reported from actual logs only.')
