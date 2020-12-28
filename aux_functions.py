import os
import re
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime
from glob import glob
import math
from copy import deepcopy

logging.basicConfig(format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%H:%M:%S]')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def extract_date(file, city):
    """
    Obtain date from file path
    Args:
    :file: path to file of data being loaded
    :city: city to which the data lodaded belongs

    Returns:
    datetime object of the data date
    """
    file_name = file.split(city)[-1]
    file_name = file_name.replace('.csv', '')
    day, year = re.findall('[0-9]+', file_name)
    month = re.findall('[a-z, A-Z]+', file_name)[0]
    date_string = f'{day} {month} {year}'
    date_object = datetime.strptime(date_string, "%d %b %Y")
    
    return date_object


def extract_date_parquet(file, city):
    """
    Obtain date from waypoint file path, expected 'utc_date=yyyy-mm-dd'
    Args:
    :file: path to file of data being loaded
    :city: city to which the data lodaded belongs

    Returns:
    datetime object of the data date
    """
    date_object = None
    file_name = file.split(city)[-1]
    res = re.findall('[0-9]+', file_name)
    if len(res) > 2:
        year, month, day = res
        date_string = f'{day} {month} {year}'
        date_object = datetime.strptime(date_string, "%d %m %Y")
    
    return date_object


def sort_df(dfs, dates):
    """
    Given two lists of dates and dfs containing data, return
    both lists sorted by dates
    Args:
    :dfs: list of dataframes containing data
    :dates: list of dates of data extracted

    Returns:
    Sorted lists
    """
    sorted_dates = sorted(dates)
    sorted_dfs = list()
    for d in sorted_dates:
        idx = dates.index(d)
        sorted_dfs.append(dfs[idx])
    return sorted_dates, sorted_dfs


def load_tide_data(city, data_path='./Tide100/home/pedro_leon/Tide100/'):
    """
    Obtain all data available for city
    Args:
    :city: city to which the data lodaded belongs
    :data_path: path to csv files containing the data

    Returns:
    :sorted_dates: sorted dates of data extracted
    :sorted_dfs: sorted dataframes of data by date
    """
    path_file = os.path.join(data_path, city)
    files = glob(os.path.join(path_file, '*.csv'))
    
    dfs = list()
    dates = list()
    for file in files:
        dfs.append(pd.read_csv(file, sep=','))
        dates.append(extract_date(file, city))
    
    sorted_dates, sorted_dfs = sort_df(dfs, dates)
    
    logger.info(f'loading data with dates: '
                 f'{[d.strftime("%m/%d/%Y") for d in sorted_dates]}')
    logger.debug(f'loading data with columns: {dfs[0].columns.values}')

    return sorted_dates, sorted_dfs


def load_waypoint_data(city, data_path='./Waypoint/home/pedro_leon/Waypoint/'):
    """
    Obtain all data available for city
    Args:
    :city: city to which the data lodaded belongs
    :data_path: path to csv files containing the data

    Returns:
    :sorted_dates: sorted dates of data extracted
    :sorted_dfs: sorted dataframes of data by date
    """
    parquet_files = list()
    dates = list()
    dir_path = os.path.join(data_path, city)
    for x in os.walk(dir_path):
        date = extract_date_parquet(x[0], city)
        if date:
            files = glob(os.path.join(x[0], '*'))
            if len(files)>0:
                parquet_files.append(files)
                dates.append(date)
    
    dfs = list()
    for idx, files in enumerate(parquet_files):
        df_list = list()
        for file in files:
            df_list.append(pd.read_parquet(file))
        dfs.append(pd.concat(df_list, ignore_index=True))
        
    logger.debug(f'Found {len(dfs)} dfs with shapes '
                 f'{[df.shape for df in dfs]}')
    
    sorted_dates, sorted_dfs = sort_df(dfs, dates)

    return sorted_dates, sorted_dfs


def parquet_to_tide_data(dfs, tile_column, features):
    """
    Data may be delivered in csv or parquet files (if no formar 
    specified, is probably a parquet).
    
    Args:
    :dfs: list of data dataframes
    :tile_column: dataframe column with tile id
    :features: columns to keep from original data (e.g. devices, records)
    
    Returns:
    Processed data as read from csv file
    """
    for idx, df in tqdm(enumerate(dfs)):
        if all(c in df.columns for c in ['ts_15', tile_column]):
            dfs[idx]['minute_of_day'] = df.ts_15.apply(lambda x: x.hour)
            group = dfs[idx].groupby([tile_column, 'minute_of_day'])[features].mean()
        elif all(c in df.columns for c in ['hour_of_day', tile_column]):
            dfs[idx]['hour_of_day'] = df.hour_of_week.apply(lambda x: x%24)        
            group = dfs[idx].groupby([tile_column, 'hour_of_day'])[features].mean()
        else:
            logger.error(f'Error while converting data. Df columns: {df.columns}')
        dfs[idx] = group.reset_index()
    
    return dfs


def get_tile_bbox(tile_id):
    """
    Returns the corner points as latitude longitude for a skyhook tile
    Created on Sep 12, 2017
    @author: pathum mudannayake

    Args:
    :tile_id: hexadecimal code for a tile, regardles the level

    Returns:
    Four points with their lat and lon coordinates
    """
    
    tile_whole_str = tile_id[0:4] #first four hex digits represent the 1x1 degree tile
    tile_frac_str = tile_id[4:] #rest correspond to the fractional portion of a coordinate
    level = len(tile_frac_str) #level is the number of hex digits that represent the fractional portion of a tile
    
    lat_frac = 0
    lon_frac = 0
    
    #hex string to int
    whole = int(tile_whole_str, 16)
    
    if level > 0:
        frac = int(tile_frac_str, 16) 
    
    #The world is divided laterally along the x axis into 360 sections. To know the latitude, divide by 360, which gives the 
    #index of the vertical partition. This is a number from 0 to 180. Negating by 90 gives the correct latitude.
    
    #Taking the modulo by 360 gives the index of the lateral partition. Negating by 180 gives the longitude value. 
    whole_lat = int(whole / 360) - 90
    whole_lon = int(whole % 360) - 180
    
    
    ##
    #Determining the fractions
    
    #Each hex digit in the fractional part represents the index of the corresponding 4x4 grid for each level, starting with
    #the highest level by the least significant hex digit 
    
    tude_shift = 0
    tile_shift = 0
    
    for _ in range(level):
        current_frac = ((frac >> tile_shift) & 15) # & with 15, i.e., 1111 to get the last four digits in the fractional part, 
                                              # each time shift by 4 digits to go to the next hex digit starting with the right most one 
        
        # the first and last two digits in each four digit block correspond to the latitude portion and the longitude portion 
        lat_frac = lat_frac + (((current_frac & 12)>>2) << tude_shift) # & by 12, 1100 to get the latitude portion, shift two digits to remove the trailing 
                                                                       # 0 bits and shift the entire thing by  
        lon_frac = lon_frac + ((current_frac & 3) << tude_shift) # similarly & by 3, 0011 to get the last two digits with correspond to the longitude fraction
                                                                 
        # each two digits derived are shifted appropriately, by tude_shift, to insert at the correct position in the 
        # binary string corresponding to either latitude or longitude 
                                                                 
        tile_shift += 4 #each hex digit is extracted by shifting the binary representation of a tile by 4 digits
        tude_shift += 2 #two binary digits are inserted from right most and two digit shifts each time the loop runs 
    
    
    div_factor = 4**level
        
    min_lat_frac = (lat_frac + 0.0)/div_factor
    min_lon_frac = (lon_frac + 0.0)/div_factor

    max_lat_frac = (lat_frac + 1.0)/div_factor
    max_lon_frac = (lon_frac + 1.0)/div_factor

    
    min_lat = whole_lat + min_lat_frac
    min_lon = whole_lon + min_lon_frac
    
    max_lat = whole_lat + max_lat_frac
    max_lon = whole_lon + max_lon_frac
    
    return ((min_lat, min_lon), (min_lat, max_lon), (max_lat, max_lon), (max_lat, min_lon))


def get_tileid(latitude, longitude, level):
    """
    SKYHOOK FUNCTION
    Get a tile id containing one point of given latitude and
    longitude
    
    Args:
    :latitude: point latitude
    :longitude: point longitude
    :level: level of the tile containing the point
    
    Returns:
    Tile hexadecimal code
    """

    init_level = level
    shift = 0

    tileid = math.trunc((16**level)*(360*math.floor(90+latitude) + math.floor(180+longitude)))    
    lat_fraction = math.trunc((4**level)*(latitude - math.floor(latitude)))
    lon_fraction = math.trunc((4**level)*(longitude - math.floor(longitude)))
    
    while level > 0:
        level = level-1;
        tileid = tileid + ((((lat_fraction & 3) << 2) + (lon_fraction & 3)) << shift);
        shift = shift + 4;
        lat_fraction = lat_fraction >> 2;
        lon_fraction = lon_fraction >> 2;

    zero_pad = init_level + 4

    return "{0:0{1}X}".format(tileid, zero_pad)


def get_tile_centroid(tile_id):
    """
    SKYHOOK FUNCTION
    Get the centroid of a tile (rectangle) given the hexadecimal code of
    the tile
    
    Args:
    :tile_id: hexadecimal code of the tile, regardless the level
    
    Returns:
    Tile centroid as a tuple (lat, lon)
    """
    
    tile_whole_str = tile_id[0:4] #first four hex digits represent the 1x1 degree tile
    tile_frac_str = tile_id[4:] #rest correspond to the fractional portion of a coordinate
    level = len(tile_frac_str) #level is the number of hex digits that represent the fractional portion of a tile
    
    lat_frac = 0
    lon_frac = 0
    
    ##
    #hex string to int
    whole = int(tile_whole_str, 16)
    frac = int(tile_frac_str, 16) 
    
    ##
    #The world is divided laterally along the x axis into 360 sections. To know the latitude, divide by 360, which gives the 
    #index of the vertical partition. This is a number from 0 to 180. Negating by 90 gives the correct latitude.
    #
    #Taking the modulo by 360 gives the index of the lateral partition. Negating by 180 gives the longitude value. 
    whole_lat = int(whole / 360) - 90;
    whole_lon = int(whole % 360) - 180;
    
    
    ##
    #Determining the fractions
    
    #Each hex digit in the fractional part represents the index of the corresponding 4x4 grid for each level, starting with
    #the highest level by the least significant hex digit 
    
    tude_shift = 0
    tile_shift = 0
    
    for i in range(level):
        current_frac = ((frac >> tile_shift) & 15) # & with 15, i.e., 1111 to get the last four digits in the fractional part, 
                                              # each time shift by 4 digits to go to the next hex digit starting with the right most one 
        
        # the first and last two digits in each four digit block correspond to the latitude portion and the longitude portion 
        lat_frac = lat_frac + (((current_frac & 12)>>2) << tude_shift) # & by 12, 1100 to get the latitude portion, shift two digits to remove the trailing 
                                                                       # 0 bits and shift the entire thing by  
        lon_frac = lon_frac + ((current_frac & 3) << tude_shift) # similarly & by 3, 0011 to get the last two digits with correspond to the longitude fraction
                                                                 
        # each two digits derived are shifted appropriately, by tude_shift, to insert at the correct position in the 
        # binary string corresponding to either latitude or longitude 
                                                                 
        tile_shift += 4 #each hex digit is extracted by shifting the binary representation of a tile by 4 digits
        tude_shift += 2 #two binary digits are inserted from right most and two digit shifts each time the loop runs 
    
    
    div_factor = 4**level
        
    cent_lat_frac = (lat_frac + 0.5)/(div_factor)
    cent_lon_frac = (lon_frac + 0.5)/(div_factor)
    
    cent_lat = whole_lat + cent_lat_frac
    cent_lon = whole_lon + cent_lon_frac
    
    return (cent_lat, cent_lon)