## Installation

Make sure python3.13 is installed on your machine.

### Mac OS
if python3.13 is not installed, run the following command:
```bash
brew install python@3.13
```

then
``` bash
alias python="python3.13"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### Windows
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```