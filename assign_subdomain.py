#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
assigns subdomains to cookies
---------------------------------------------------------
IN

---------------------------------------------------------
OUT

---------------------------------------------------------
EXAMPLE CALL

python /home/kbrennan/cookie_cutter/assign_subdomain.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/grids/subdomains.nc

---------------------------------------------------------
Killian P. Brennan
06.05.2024
---------------------------------------------------------
"""

import os
import argparse
import xarray as xr


def main(cookie_dir, subdomain_dir):

    subdomains = xr.open_dataset(subdomain_dir)
    subdomains = subdomains.drop(["time_bnds", "rotated_pole"])
    subdomains = subdomains.squeeze()

    os.makedirs(os.path.join(cookie_dir,'subdomains'), exist_ok=True)
    keys = list(subdomains.keys())
    for k in keys:
        os.makedirs(os.path.join(cookie_dir,'subdomains', k), exist_ok=True)

    cookie_files = os.listdir(cookie_dir)

    # must end in .nc
    cookie_files = [f for f in cookie_files if f.endswith(".nc")]

    cookie_files = sorted(cookie_files)

    for f in cookie_files:
        cookie = xr.open_dataset(os.path.join(cookie_dir, f))
        cookie = cookie.squeeze()
        rlon = cookie.rlon.sel(x=0, y=0).values
        rlat = cookie.rlat.sel(x=0, y=0).values
        for dom in subdomains.keys():
            test = (
                subdomains[dom]
                .sel(
                    rlon=rlon,
                    rlat=rlat,
                    method="nearest",
                )
                .values
            )
            if test == 1.0:
                print(f"Assigning {f} to {dom}")
                os.symlink(
                    os.path.join(cookie_dir, f),
                    os.path.join(cookie_dir, 'subdomains', dom, f),
                )
    return


def determine_subdomain(cookie, subdomains):
    """figure out which subdomain each cookie belongs to"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composites from cookies")
    parser.add_argument("cookie_dir", type=str, help="Directory containing cookies")
    parser.add_argument("subdomain_dir", type=str, help="Directory of subdomain file")
    args = parser.parse_args()
    main(args.cookie_dir, args.subdomain_dir)
