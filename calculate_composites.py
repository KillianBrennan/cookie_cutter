import os

import xarray as xr
import numpy as np

import argparse


def main(cookie_dir):
    os.makedirs('composites', exist_ok=True)

    subdomains = xr.open_dataset('/home/kbrennan/phd/data/climate/grids/subdomains_rot.nc')
    subdomains = subdomains.drop(['time_bnds','rotated_pole'])
    subdomains = subdomains.squeeze()

    cookies = load_cookies(cookie_dir)

    return

def load_cookies(cookie_dir):
    cookies = xr.open_mfdataset(os.path.join(cookie_dir, '*.nc'))
    return cookies

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate composites from cookies')
    parser.add_argument('cookie_dir', type=str, help='Directory containing cookies')
    args = parser.parse_args()
    main(args.cookie_dir)