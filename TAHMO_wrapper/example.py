"""Examples for the TAHMO DataHub API wrapper.

Run this file from the project root (the folder that contains the TAHMO folder).

Credentials are taken from --key/--secret, then from the environment variables
TAHMO_API_KEY and TAHMO_API_SECRET, and otherwise you are asked for them
(the secret is typed hidden).

Examples
--------
List the available variables and stations:
    python example.py --list-variables
    python example.py --list-stations

One station, two variables, saved as CSV:
    python example.py --stations TA00001 --start 2024-01-01 --end 2024-01-31 \
        --vars pr te --format csv

Several stations, saved as Parquet:
    python example.py --stations TA00001 TA00002 --start 2024-01-01 --end 2024-06-30 \
        --vars pr te --format parquet

All stations inside a bounding box (LAT_MIN LAT_MAX LON_MIN LON_MAX), saved as Zarr:
    python example.py --bbox -4.7 5.0 33.9 41.9 --start 2024-01-01 --end 2024-03-31 \
        --vars pr --format zarr --output output/kenya
"""

import argparse
import getpass
import os

from TAHMO import apiWrapper, SUPPORTED_FORMATS


# ----------------------------------------------------------------------
# Programmatic examples (the same calls the command line below makes)
# ----------------------------------------------------------------------
def example_single_station(api):
    """Download one station and save it as CSV."""
    df = api.getMeasurements('TA00001',
                             startDate='2024-01-01',
                             endDate='2024-01-31',
                             variables=['pr', 'te'])
    print(df.head())
    api.saveData(df, 'output/ta00001.csv')


def example_bounding_box(api):
    """Download every station inside a latitude/longitude box and save as Parquet."""
    df = api.getMeasurementsInBounds(latMin=-4.7, latMax=5.0,
                                     lonMin=33.9, lonMax=41.9,
                                     startDate='2024-01-01',
                                     endDate='2024-01-31',
                                     variables=['pr'])
    print(df.shape)
    api.saveData(df, 'output/kenya_rainfall.parquet')


def example_raw_data(api):
    """Download the unfiltered (raw) dataset instead of the quality controlled one."""
    return api.getRawMeasurements('TA00001', '2024-01-01', '2024-01-07', ['te'])


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------
def get_credentials(args):
    key = args.key or os.environ.get('TAHMO_API_KEY') or input('API key: ').strip()
    secret = args.secret or os.environ.get('TAHMO_API_SECRET') or getpass.getpass('API secret: ')
    return key, secret


def build_parser():
    p = argparse.ArgumentParser(description='Download TAHMO station data.')
    p.add_argument('--key', help='API key (or set TAHMO_API_KEY)')
    p.add_argument('--secret', help='API secret (or set TAHMO_API_SECRET; leave out to be prompted)')
    p.add_argument('--start', help='Start date, e.g. 2024-01-01')
    p.add_argument('--end', help='End date, e.g. 2024-06-30 (the whole day is included)')
    p.add_argument('--vars', nargs='+', default=None, metavar='VAR',
                   help='Variable shortcodes, e.g. pr te. Default: all variables')
    p.add_argument('--stations', nargs='+', metavar='CODE', help='One or more station codes')
    p.add_argument('--bbox', nargs=4, type=float,
                   metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                   help='Bounding box. Downloads every station inside it')
    p.add_argument('--dataset', choices=['controlled', 'raw'], default='controlled')
    p.add_argument('--format', choices=SUPPORTED_FORMATS, default='csv', help='Output format')
    p.add_argument('--output', default='output/measurements',
                   help='Output path without extension (default: output/measurements)')
    p.add_argument('--list-variables', action='store_true', help='Print the available variables and exit')
    p.add_argument('--list-stations', action='store_true', help='Print the available stations and exit')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    api = apiWrapper()
    api.setCredentials(*get_credentials(args))

    if args.list_variables:
        for shortcode, variable in api.getVariables().items():
            print('%-8s %s' % (shortcode, variable.get('description', '')))
        return

    if args.list_stations:
        for code, station in api.getStations().items():
            print('%-10s %s' % (code, station.get('location', {}) if isinstance(station.get('location'), dict) else ''))
        return

    if not args.start or not args.end:
        parser.error('--start and --end are required')
    if not args.stations and not args.bbox:
        parser.error('give --stations and/or --bbox')

    codes = list(args.stations or [])
    if args.bbox:
        in_box = api.getStationsInBounds(*args.bbox)
        print('%d stations found in the bounding box' % len(in_box))
        codes += [c for c in in_box if c not in codes]

    df = api.getMeasurementsForStations(codes, args.start, args.end,
                                        variables=args.vars, dataset=args.dataset)
    if df.empty:
        print('No data returned for these stations and dates.')
        return

    print(df.head())
    print('Shape: %s rows x %s columns' % df.shape)
    if api.lastErrors:
        print('Failed stations: %s' % ', '.join(api.lastErrors))

    api.saveData(df, args.output, fmt=args.format)


if __name__ == '__main__':
    main()
