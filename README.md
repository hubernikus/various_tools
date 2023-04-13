# Various Tools Library
---
[Badge License]: https://img.shields.io/badge/License-MPL_2.0-FF7139.svg?style=for-the-badge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
---

Various tools from linear algebra to vector spaces to simplify and speed up computation.

It consists of three main sub-libraries:
- Dynamical Systems
- Directional Space
- Angle Math
- Linear Algebra Helpers
- General Math / Logic

## Custom Environment
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >3.10).

``` bash
python3.10 -m venv .venv
```
with python -V > 3.10

Activate your environment
``` sh
source .venv/bin/activate
```


## Setup / Install
(using custom pip-environment)
``` bash
pip install -r requirements.txt && pip install -e .
```

In order to be able to save animations from `animator.py` you need `ffmpeg` installed. For linux (Ubuntu) this can be done via:
``` bash
sudo apt install ffmpeg
```

## Usage


# Unit Testing
In order to run all the test-scripts, run in the command line in the main folder:
``` bash
pytest
```

