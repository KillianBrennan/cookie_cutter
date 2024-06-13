import os

import xarray as xr
import numpy as np

import argparse


def main(cookie_dir):
    os.makedirs("composites", exist_ok=True)

    subdomain_dir = os.path.join(cookie_dir, "subdomains")

    for dom in os.listdir(subdomain_dir):
        if os.path.isdir(os.path.join(subdomain_dir, dom)):
            print(dom)
            cookies = load_cookies(os.path.join(subdomain_dir, dom))
            mean = calculate_composite(cookies)
            mean.to_netcdf(os.path.join("composites", f"{dom}_mean.nc"))
    return


def load_cookies(cookie_dir):
    cookies = xr.open_mfdataset(
        os.path.join(cookie_dir, "*.nc"), combine="by_coords", parallel=True
    )
    return cookies


def calculate_composite(cookies):
    n_cookies = cookies.cookie_id.shape[0]
    n_fields = cookies.T.count(dim='cookie_id')

    mean = cookies.mean(dim="cookie_id", skipna=True)
    mean = mean.compute()
    mean['n_cookies'] = n_cookies
    mean['n_cookies'].attrs = {
        'long_name': 'number of cookies in composite'
    }
    mean['n_fields'] = n_fields 
    mean['n_fields'].attrs = {
        'long_name': 'number of fields in composite'
    }
    return mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composites from cookies")
    parser.add_argument("cookie_dir", type=str, help="Directory containing cookies")
    args = parser.parse_args()
    main(args.cookie_dir)
