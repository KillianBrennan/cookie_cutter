#!/bin/bash
set -x

source /users/kbrennan/cookie_cutter/venv_cookie/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
python extract_environment_paralell_years.py present