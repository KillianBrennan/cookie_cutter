#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run extract_environment.py in parallel for all years
"""

import os
import subprocess
import sys
from multiprocessing import Pool
import numpy as np

import extract_environment as ee
from argparse import ArgumentParser

def main(period):
    # period can either be present or future
    # check if period is valid
    if period not in ['present', 'future']:
        print('Invalid period. Exiting...')
        sys.exit(1)

    if period == 'present':
        years = np.arange(2011, 2022)
        with Pool(12) as p:
            p.map(cut_year_present, years)


    if period == 'future':
        years = np.arange(2085, 2096)
        with Pool(12) as p:
            p.map(cut_year_future, years)

    return

def cut_year_present(year):
    envdir = '/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f'
    trackdir = '/store/c2sm/scclim/climate_simulations/present_day/hail_tracks/filtered'
    outpath = '/users/kbrennan/scratch/cookies/present'

    ee.main(trackdir, envdir, outpath, str(year)+'0101', str(year+1)+'0101')
    return

def cut_year_future(year):
    envdir = '/project/pr133/irist/scClim/RUN_2km_cosmo6_climate_PGW_MPI_HR/output/lm_f/'
    trackdir = '/store/c2sm/scclim/climate_simulations/PGW_MPI_HR/hail_tracks'
    outpath = '/users/kbrennan/scratch/cookies/future'

    ee.main(trackdir, envdir, outpath, str(year)+'0101', str(year+1)+'0101')
    return
    

if __name__ == '__main__':
    p = ArgumentParser()
    
    # period can either be present or future
    p.add_argument('period', type=str)

    args = p.parse_args()
    main(args.period)

