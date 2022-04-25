python3 -m venv endoscopy-venv
source  endoscopy-venv/bin/activate
mkdir temp
TMPDIR=temp pip install -r requirements.txt
rm temp
deactivate