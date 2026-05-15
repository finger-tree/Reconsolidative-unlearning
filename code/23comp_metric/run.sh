# pyenv install 3.10.14   # if not already installed
# pyenv local 3.10.14     # in the project directory
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# python launch.py


rm -rf .venv
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/google-deepmind/xmanager.git
# xmanager launch.py
# # python launch.py


python -m main \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints \
  --output_dir ./outputs

  