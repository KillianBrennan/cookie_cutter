# cookie_cutter

setup venv
    make sure you've set up a virtual environment with the necessary packages in the cookie_cutter directory:
    either name it venv_cookie or change the path in the run scripts

active venv
    source /users/kbrennan/cookie_cutter/venv_cookie/bin/activate

test cookie cutter on daint in login node as test:
    python /users/kbrennan/cookie_cutter/extract_environment.py /users/kbrennan/climate/tracks/post-processing/present_day/hail_tracks /users/kbrennan/climate/present /users/kbrennan/scratch/cookies/present 20110101 20110201

test parallel script in login node:
    python extract_environment_paralell_years.py present

run years in parallel as jobs:
    sh run_present.sh
    sh run_future.sh