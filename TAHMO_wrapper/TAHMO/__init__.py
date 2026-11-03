"""TAHMO DataHub API wrapper.

Download weather station data from the TAHMO DataHub (https://datahub.tahmo.org)
as pandas DataFrames, filter stations by a latitude/longitude bounding box,
and save the result as CSV, Parquet or Zarr.

Quick start::

    from TAHMO import apiWrapper

    api = apiWrapper()
    api.setCredentials('MY_KEY', 'MY_SECRET')
    df = api.getMeasurements('TA00001', '2024-01-01', '2024-01-31', ['pr', 'te'])
    api.saveData(df, 'output/ta00001.parquet')
"""

# Module dependencies
import datetime  # noqa: F401  (kept for backwards compatibility)
import gc
import os

import dateutil.parser
import numpy as np
import pandas as pd
import requests
import requests.auth

__all__ = ['apiWrapper', 'API_BASE_URL', 'SUPPORTED_FORMATS']

# Module Constants
API_BASE_URL = 'https://datahub.tahmo.org'
API_MAX_PERIOD_DAYS = 365           # The API accepts at most ~1 year per request.
DATASETS = ('controlled', 'raw')
SUPPORTED_FORMATS = ('csv', 'parquet', 'zarr')

# Environment variables that can hold the credentials.
ENV_KEY = 'TAHMO_API_KEY'
ENV_SECRET = 'TAHMO_API_SECRET'


class apiWrapper(object):
    """Client for the TAHMO DataHub API.

    Credentials can be given to the constructor, set later with
    ``setCredentials`` or read from the ``TAHMO_API_KEY`` and
    ``TAHMO_API_SECRET`` environment variables.
    """

    def __init__(self, key=None, secret=None):
        self.apiKey = key or os.environ.get(ENV_KEY, '')
        self.apiSecret = secret or os.environ.get(ENV_SECRET, '')
        self.lastErrors = {}   # station -> error message of the last multi-station download

    # ------------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------------
    def setCredentials(self, key, secret):
        self.apiKey = key
        self.apiSecret = secret

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------
    def getMeasurements(self, station, startDate=None, endDate=None, variables=None, dataset='controlled'):
        """Download the measurements of ONE station.

        station    -- station code, e.g. 'TA00001'
        startDate  -- start of the period, e.g. '2024-01-01' (required)
        endDate    -- end of the period, e.g. '2024-06-30' (required).
                      A date without a time is treated as the END of that day.
        variables  -- list of variable shortcodes, e.g. ['pr', 'te'] (default: all)
        dataset    -- 'controlled' (quality checked) or 'raw'

        Returns a DataFrame indexed by time with one column per variable
        (or ``variable_sensor`` when several sensors measure the same variable).
        Observations whose quality flag is not 1 are returned as NaN.
        """
        if dataset not in DATASETS:
            raise ValueError("dataset must be one of %s" % (DATASETS,))
        variables = self.__normaliseVariables(variables)

        endpoint = 'services/measurements/v2/stations/%s/measurements/%s' % (station, dataset)
        dateSplit = self.__splitDateRange(startDate, endDate)
        seriesHolder = {}

        for _, row in dateSplit.iterrows():
            params = {'start': row['start'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                      'end': row['end'].strftime('%Y-%m-%dT%H:%M:%SZ')}
            if variables and len(variables) == 1:
                params['variable'] = variables[0]

            response = self.__request(endpoint, params)
            self.__collectObservations(response, variables, seriesHolder)

            # Free memory, responses can be large.
            del response
            gc.collect()

        return self.__buildDataFrame(seriesHolder)

    def getRawMeasurements(self, station, startDate=None, endDate=None, variables=None):
        return self.getMeasurements(station, startDate=startDate, endDate=endDate,
                                    variables=variables, dataset='raw')

    def getMeasurementsForStations(self, stations, startDate, endDate, variables=None,
                                   dataset='controlled', skipErrors=True):
        """Download the measurements of SEVERAL stations into one DataFrame.

        The columns are a MultiIndex of (station, variable). Stations without
        data are left out. When ``skipErrors`` is True, a station that fails
        is skipped and its error is stored in ``self.lastErrors``.
        """
        if isinstance(stations, str):
            stations = [stations]

        frames = {}
        self.lastErrors = {}
        for number, code in enumerate(stations, start=1):
            print('Station %s (%d/%d)' % (code, number, len(stations)))
            try:
                df = self.getMeasurements(code, startDate=startDate, endDate=endDate,
                                          variables=variables, dataset=dataset)
            except Exception as error:
                if not skipErrors:
                    raise
                self.lastErrors[code] = str(error)
                print('  skipped %s: %s' % (code, error))
                continue
            if not df.empty:
                frames[code] = df

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1, sort=True)

    def getMeasurementsInBounds(self, latMin, latMax, lonMin, lonMax, startDate, endDate,
                                variables=None, dataset='controlled', skipErrors=True):
        """Download the measurements of all stations inside a bounding box."""
        stations = self.getStationsInBounds(latMin, latMax, lonMin, lonMax)
        print('%d stations found in the bounding box' % len(stations))
        return self.getMeasurementsForStations(list(stations), startDate, endDate,
                                               variables=variables, dataset=dataset,
                                               skipErrors=skipErrors)

    # ------------------------------------------------------------------
    # Stations and variables
    # ------------------------------------------------------------------
    def getStations(self):
        """Return a dict with all stations, keyed by station code."""
        response = self.__request('services/assets/v2/stations', {'sort': 'code'})
        stations = {}
        if 'data' in response and isinstance(response['data'], list):
            for element in response['data']:
                stations[element['code']] = element
        return stations

    def getStationsInBounds(self, latMin, latMax, lonMin, lonMax):
        """Return the stations whose coordinates are inside the bounding box.

        Latitude runs from -90 to 90 and longitude from -180 to 180.
        Stations without coordinates are ignored.
        """
        if latMin > latMax or lonMin > lonMax:
            raise ValueError('Bounds must be given as latMin <= latMax and lonMin <= lonMax')

        selected = {}
        for code, station in self.getStations().items():
            coordinates = self._stationCoordinates(station)
            if coordinates is None:
                continue
            lat, lon = coordinates
            if latMin <= lat <= latMax and lonMin <= lon <= lonMax:
                selected[code] = station
        return selected

    def getVariables(self):
        """Return a dict with all variables, keyed by shortcode."""
        response = self.__request('services/assets/v2/variables', {})
        variables = {}
        if 'data' in response and isinstance(response['data'], list):
            for element in response['data']:
                variables[element['variable']['shortcode']] = element['variable']
        return variables

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    @staticmethod
    def saveData(df, path, fmt=None):
        """Save a DataFrame as CSV, Parquet or Zarr and return the path written.

        path -- target path. The format is taken from the extension
                (.csv, .parquet, .zarr) unless ``fmt`` is given. If ``fmt`` is
                given and the path has no matching extension, it is added.
        fmt  -- 'csv', 'parquet' or 'zarr'

        Zarr is stored as a folder, not a single file.
        """
        if df is None or df.empty:
            raise ValueError('There is no data to save')

        root, extension = os.path.splitext(path)
        if fmt is None:
            fmt = extension.lstrip('.').lower()
        fmt = fmt.lower()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError('Unsupported format %r, choose one of %s' % (fmt, SUPPORTED_FORMATS))
        if extension.lstrip('.').lower() != fmt:
            path = '%s.%s' % (path, fmt)

        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        out = df.copy()
        # Parquet and Zarr need flat string column names, not (station, variable) tuples.
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = ['%s_%s' % pair for pair in out.columns]
        out.columns = [str(column) for column in out.columns]
        out.index.name = out.index.name or 'time'

        if fmt == 'csv':
            out.to_csv(path)
        elif fmt == 'parquet':
            out.to_parquet(path)       # needs pyarrow
        else:
            # Zarr (via xarray) wants unique, timezone naive (UTC) timestamps.
            if getattr(out.index, 'tz', None) is not None:
                out.index = out.index.tz_convert(None)
            out = out[~out.index.duplicated(keep='first')]
            out.to_xarray().to_zarr(path, mode='w')   # needs xarray and zarr

        print('Saved %s' % path)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _stationCoordinates(station):
        """Return (latitude, longitude) of a station dict, or None if unknown."""
        location = station.get('location')
        source = location if isinstance(location, dict) else station
        lat = source.get('latitude', source.get('lat'))
        lon = source.get('longitude', source.get('lon', source.get('lng')))
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def __normaliseVariables(variables):
        if variables is None:
            return None
        if isinstance(variables, str):
            return [variables]
        return list(variables)

    def __request(self, endpoint, params):
        print('API request: %s' % endpoint)
        apiRequest = requests.get(
            '%s/%s' % (API_BASE_URL, endpoint),
            params=params,
            auth=requests.auth.HTTPBasicAuth(
                self.apiKey,
                self.apiSecret
            )
        )

        if apiRequest.status_code == 200:
            return apiRequest.json()
        else:
            return self.__handleApiError(apiRequest)

    def __collectObservations(self, response, variables, seriesHolder):
        """Add the observations of one API response to ``seriesHolder``.

        seriesHolder maps variable shortcode -> list of [time, value, sensor, quality].
        """
        if not isinstance(response, dict):
            return

        for result in response.get('results', []):
            for serie in result.get('series', []):
                if 'values' not in serie:
                    continue

                columns = serie['columns']
                observations = serie['values']

                timeIndex = columns.index('time')
                qualityIndex = columns.index('quality')
                variableIndex = columns.index('variable')
                sensorIndex = columns.index('sensor')
                valueIndex = columns.index('value')

                # Variables requested, or all variables found in the observations.
                if not variables:
                    shortcodes = list(set(x[variableIndex] for x in observations))
                else:
                    shortcodes = variables

                for shortcode in shortcodes:
                    # Format of each element: [time, value, sensor, quality].
                    # The value is only kept when the quality flag is 1.
                    timeserie = [[x[timeIndex],
                                  x[valueIndex] if x[qualityIndex] == 1 else np.nan,
                                  x[sensorIndex],
                                  x[qualityIndex]]
                                 for x in observations if x[variableIndex] == shortcode]
                    seriesHolder.setdefault(shortcode, []).extend(timeserie)

    def __buildDataFrame(self, seriesHolder):
        """Turn the collected observations into one DataFrame."""
        series = []

        for shortcode in seriesHolder:
            timestamps = [x[0] for x in seriesHolder[shortcode]]

            if len(timestamps) > len(set(timestamps)):
                # Duplicate timestamps: several sensors measure this variable, split them.
                print('Split observations for %s per sensor' % shortcode)
                sensors = list(set(x[2] for x in seriesHolder[shortcode]))
                for sensor in sensors:
                    sensorSerie = [x for x in seriesHolder[shortcode] if x[2] == sensor]
                    index = pd.DatetimeIndex([pd.Timestamp(x[0]) for x in sensorSerie])
                    serie = pd.Series([x[1] for x in sensorSerie], index=index)
                    series.append(serie.to_frame('%s_%s' % (shortcode, sensor)))
            else:
                values = [x[1] for x in seriesHolder[shortcode]]
                if len(values) > 0:
                    serie = pd.Series(values, index=pd.DatetimeIndex(timestamps))
                    series.append(serie.to_frame(shortcode))

            gc.collect()

        if len(series) > 0:
            df = pd.concat(series, axis=1, sort=True)
        else:
            df = pd.DataFrame()

        del series
        gc.collect()
        return df

    @staticmethod
    def __toUtcTimestamp(value, name):
        """Parse a date (string, datetime or Timestamp) to a naive UTC Timestamp."""
        if value is None:
            raise ValueError('%s is required' % name)
        try:
            if isinstance(value, str):
                timestamp = pd.Timestamp(dateutil.parser.parse(value))
            else:
                timestamp = pd.Timestamp(value)
        except (ValueError, OverflowError, TypeError):
            raise ValueError('Invalid %s: %r' % (name, value))

        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert('UTC').tz_localize(None)

        # A date without a time as END date means "until the end of that day".
        if name == 'endDate' and isinstance(value, str) and ':' not in value:
            timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return timestamp

    # Split date range into intervals of at most 365 days.
    def __splitDateRange(self, inputStartDate, inputEndDate):
        start = self.__toUtcTimestamp(inputStartDate, 'startDate')
        end = self.__toUtcTimestamp(inputEndDate, 'endDate')
        if start >= end:
            raise ValueError('startDate must be before endDate')

        period = pd.Timedelta(days=API_MAX_PERIOD_DAYS)
        rows = []
        current = start
        while current < end:
            following = current + period
            if following >= end:
                rows.append((current, end))
                break
            rows.append((current, following - pd.Timedelta(seconds=1)))
            current = following

        return pd.DataFrame(rows, columns=['start', 'end'])

    def __handleApiError(self, apiRequest):
        json = None
        try:
            # Try to parse json and check if body contains a specific error message.
            json = apiRequest.json()
        except ValueError:
            pass

        if json and 'error' in json and 'message' in json['error']:
            print(json)
            raise Exception(json['error']['message'])
        raise Exception('API request failed with status code %s' % apiRequest.status_code)
