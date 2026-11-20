# FYS-STK4155 - Project 3

**Authors:** Giacomo Saporetti, Amanda Rae Josephine, Narne Abhinav Chowdhary

## Overview


## Repository contents

| File / folder                 | Description                                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `Code`                        | Main directory with all code, results and plots                                                                                            |
| `doc`                         | LaTeX source of the report and corresponding PDF file                                                                                      |
| `Code/Project3.ipynb`         | Main Jupyter notebook |

| `Code/results`                | Cached results (CSV and joblib files) created by the notebook so that long hyperparameter scans are not recomputed. Delete the files to rerun the scans |
| `Code/figures`                | Resulting graphs                                                                                                                           |

## Requirements

- Python 3.10 or newer
- Jupyter (Notebook or JupyterLab)
- Python packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `joblib`
  - `jax` (only needed for the gradient check against automatic differentiation)
- The MNIST data set as a single NumPy archive, `mnist.npz` (about 11 MB and already present in the directory).

## Setup

Clone the repository:

```bash
git clone https://github.com/AmandaRaeJ16/FYS-STK4155-Project2
cd FYS-STK4155-Project2
```

Create a virtual environment (recommended) and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Running the code

Start Jupyter from the `Code` folder and open the notebook:

```bash
cd Code
jupyter notebook Project3.ipynb
```

## Building the report (optional)

To compile the report from the LaTeX source:

```bash
cd doc
pdflatex Report.tex
```

Run it twice if the references or table of contents are not updated. Make sure the figures
in `Code/figures` are present first.

