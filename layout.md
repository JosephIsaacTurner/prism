# Project Layout

This document describes the structure of the `pstn` project.

## Root Directory (`pstn/`)

The root directory contains the main components of the project, including code, notebooks, tests, and essential project files.

```
pstn/
├── pstn/              # Core Python package
│   ├── stats.py       # Statistical functions
│   ├── inference.py   # Inference functions (hypothesis testing classes, etc.)
│   ├── loading.py     # Functions to load data
│   ├── plotting.py    # Functions to plot data
│   ├── data/          # Directory for data files (brain image templates, etc.)
├── notebooks/         # Jupyter notebooks for experimentation & analysis
├── tests/             # Unit and integration tests
├── assets/            # Images and other assets for documentation
├── manuscript/        # Manuscript outlining background and methodology
├── layout.md          # Explanation of the layout of the project
├── README.md          # Introduction to the project for new users
├── requirements.txt   # List of dependencies
└── .gitignore         # Files and directories to be ignored by Git
```

## Directories:

- **`pstn/`**: The core Python package containing all source code.
- **`notebooks/`**: Jupyter notebooks for experimentation, documentation, and analysis.
- **`tests/`**: Unit and integration tests for the package.

## Files:

- **`layout.md`**: This file, explaining the project layout.
- **`README.md`**: Introduction to the project for new users.
- **`requirements.txt`**: Lists dependencies required to run the project.
- **`.gitignore`**: Specifies files and directories that should be ignored by Git.

---

## Python Package (`pstn/pstn/`)

The core functionality of the project is implemented in the `pstn` package, which is structured as follows:

### Modules:

- **`stats.py`**: Statistical functions.
- **`inference.py`**: Functions for statistical inference, including hypothesis testing.
- **`loading.py`**: Functions for loading data.
- **`plotting.py`**: Functions for visualizing data.

### Subdirectories:

- **`data/`**: Contains data files, such as brain image templates.

### Note:
We **do not** include a `utils.py` file, as it is too vague and does not clearly define a purpose.

---

This layout ensures modularity, clarity, and maintainability for the `pstn` project.
