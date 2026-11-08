"""Run EDA from a fresh project kernel and preserve completed notebook outputs."""

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
# Keep Jupyter connection files inside the workspace on restricted Windows hosts.
runtime = root / ".tmp" / "jupyter"
runtime.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
os.environ["IPYTHONDIR"] = str(root / ".tmp" / "ipython")
os.environ["MPLCONFIGDIR"] = str(root / ".tmp" / "matplotlib")
notebook_path = root / "notebooks/main.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
client = NotebookClient(
    notebook, km=manager, timeout=1200, resources={"metadata": {"path": str(root)}}
)
try:
    client.execute(cleanup_kc=True)
finally:
    nbformat.write(notebook, notebook_path)
print("Executed all cells in a fresh project kernel:", notebook_path)
