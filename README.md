# pioneer.common

pioneer.common is a python library regrouping all the utilities of the team Pioneer.

## Installation

Before installing, you should add to your pip.conf file the gitlab pypi server url to trust.

```conf
[global]
extra-index-url = https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple
trusted-host = svleddar-gitlab.leddartech.local
```

Use the package manager [pip](https://pioneer:yK6RUkhUCNHg3e1yxGT4@svleddar-gitlab.leddartech.local/api/v4/projects/481/packages/pypi/simple/pioneer-common) to install pioneer.common .

```bash
pip install pioneer-common
```

When developing, you can link the repository to your python site-packages and enable hot-reloading of the package.
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

