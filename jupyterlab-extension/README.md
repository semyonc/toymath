# ToyMath JupyterLab extension

The extension provides native completion for `model!` and keeps the notebook
toolbar title synchronized with the notebook-local model.

Build it from the repository root:

```bash
source .venv/bin/activate
cd jupyterlab-extension
npm install
npm run build
cd ..
```

`npm run build` compiles the TypeScript source and writes the prebuilt bundle
to `labextension/`. Install that bundle into the active Jupyter environment
through the ToyMath Python package:

```bash
uv pip install --reinstall --no-deps .
jupyter labextension list
```

The extension is ready when the list contains
`@toymath/model-ui ... enabled OK`. Restart the JupyterLab server and reload
the browser tab after installing or rebuilding it; rebuilding JupyterLab
itself is not required for this prebuilt extension.
