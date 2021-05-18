jupyter-widget-example
===============================

A Custom Jupyter Widget Library

Installation
------------

To install use pip:

    $ pip install jupyter_widget_example

For a development installation (requires [Node.js](https://nodejs.org) and [Yarn version 1](https://classic.yarnpkg.com/)),

    $ git clone https://github.com/datacanvas/jupyter-widget-example.git
    $ cd jupyter-widget-example
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --overwrite --sys-prefix jupyter_widget_example
    $ jupyter nbextension enable --py --sys-prefix jupyter_widget_example

When actively developing your extension for JupyterLab, run the command:

    $ jupyter labextension develop --overwrite jupyter_widget_example

Then you need to rebuild the JS when you make a code change:

    $ cd js
    $ yarn run build

You then need to refresh the JupyterLab page when your javascript changes.
