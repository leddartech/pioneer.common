# pioneer.common

pioneer.common is a python library regrouping all the utilities of the team Pioneer.

## Installation

Before installing, you should add to your pip.conf file the gitlab pypi server url to trust.

```conf
[global]
extra-index-url = https://__token__:qcnZ-LPju8cqtpG1cpss@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple
trusted-host = svleddar-gitlab.leddartech.local
```

Use the package manager [pip](https://__token__:<your_personal_token>@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple) to install pioneer.common .

```bash
pip install pioneer-common --index-url https://__token__:<your_personal_token>@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple --trusted-host svleddar-gitlab.leddartech.local
```

Enable hot-reloading of packages during development.
```bash
python3 setup.py develop --user
```

If you don't want to install all the dependencies on your computer, you can run it in a virtual environment
```bash
pipenv install

pipenv shell
```

## Usage

```python
from pioneer.common import platform

platform.parse_datasource_name('pixell_bfc_ech')
```

