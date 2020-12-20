import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import re
import pickle

def read_heatspots_dataset(path='./data/heatspots_states_1998-2017.csv'):
    
    with open('./data/misc/months.txt', encoding='utf-8') as months:
        months_map = dict(map(lambda x: (x[1].replace('\n',''), x[0]), enumerate(months.readlines(), 1)))

    codes_map = pickle.load(open('./data/misc/st_codes.pkl', 'rb'))

    data = pd.read_csv(path, encoding='utf-8', sep='\t', dtype={'Number':'str'}) \
        .assign(Number = lambda df: df['Number'].apply(lambda x: int(x.replace('.','')))) \
        .assign(St_codes = lambda df: df['State'].map(lambda x: codes_map[x.upper()])) \
        .replace({'Month':months_map}) \
        .drop('Period', axis=1) \
        .assign(Date = lambda df: pd.PeriodIndex(year=df['Year'], month=df['Month'], freq='M')) \
        .set_index('Date')
    
    return data.sort_index().drop_duplicates()

def read_deforestation_dataset():
    data = pd.read_csv('./data/deforestation.csv')
    data.columns = data.columns.map(lambda x: x.capitalize())
    return data

def import_table(path, period=None):
    """this function import the contents of each climate station and outputs and dictionary
    with leys corresponding to properties of the importated data."""
    
    feature_names = {'DIRECAO PREDOMINANTE DO VENTO, MENSAL(° (gr))': 'windDirection',
    'EVAPORACAO DO PICHE, MENSAL(mm)': 'evaptransPiche',
    'EVAPOTRANSPIRACAO POTENCIAL, BH MENSAL(mm)': 'evaptransPot',
    'EVAPOTRANSPIRACAO REAL, BH MENSAL(mm)': 'evaptransReal',
    'INSOLACAO TOTAL, MENSAL(h)': 'insolationTotal',
    'NEBULOSIDADE, MEDIA MENSAL(décimos)': 'cloudiness',
    'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL(número)': 'precipitationDays',
    'PRECIPITACAO TOTAL, MENSAL(mm)': 'precipitationTotal',
    'PRESSAO ATMOSFERICA AO NIVEL DO MAR, MEDIA MENSAL(mB)': 'avgAtmPressureSL',
    'PRESSAO ATMOSFERICA, MEDIA MENSAL(mB)': 'avgAtmPressure',
    'TEMPERATURA MAXIMA MEDIA, MENSAL(°C)': 'avgMaxTemp',
    'TEMPERATURA MEDIA COMPENSADA, MENSAL(°C)': 'avgTemp',
    'TEMPERATURA MINIMA MEDIA, MENSAL(°C)': 'avgMinTemp',
    'UMIDADE RELATIVA DO AR, MEDIA MENSAL(%)': 'avgRelHumidity',
    'VENTO, VELOCIDADE MAXIMA MENSAL(m/s)': 'maxWindSpeed',
    'VENTO, VELOCIDADE MEDIA MENSAL(m/s)': 'avgWindSpeed',
    'VISIBILIDADE, MEDIA MENSAL(codigo)': 'visibility',
    'Unnamed: 17': 'avgAtmPressureSL',
    'Data Medicao': 'Date'}
    
    data = {}
    with open(path, 'r+', encoding='cp1252') as file:
        table_found = False
        for counter, line in enumerate(file):
            parsed_line = _parse_line_table(line)
            if isinstance(parsed_line, dict):
                data.update(parsed_line)
            elif parsed_line:
                skiprows = counter
                table_found = parsed_line 
        print('{} read...'.format(path), end=' ')
        
    if table_found:
        data['Data'] = pd.read_csv(path, sep=';', encoding='utf-8', skiprows=skiprows, index_col=0, parse_dates=[0], decimal=',', usecols=range(18))
        data['Data'] = data['Data'].rename(feature_names, axis=1)
        if period:
            data['Data'].index = data['Data'].index.to_period(freq='M')
            data['Data'].index.name = feature_names.get(data['Data'].index.name)
        print("Data collected successifully")
    else:
        print('')
    
    return data

def read_states_geometry(path):
    """Read multiple geometries into list of GeoSeries."""
    geos = []
    for file in glob.glob(path):
        geoseries = gpd.read_file(file, crs=4326)
        geoseries['geometry'] = geoseries['geometry'].apply(lambda x: Polygon(x.coords))
        geos.append(geoseries)
    
    return geos


def _parse_line_table(string):
    """this function parses the content of a string into predetermined kinds."""

    if re.match(r'Codigo Estacao', string):
        value = convert_to_number(string.replace('\n',''), type=str)
        return {'Code': str(value)}
    
    elif re.match(r'Latitude', string):
        value = convert_to_number(string.replace('\n',''))
        return {'Latitude': value}

    elif re.match(r'Longitude', string):
        value = convert_to_number(string.replace('\n',''))
        return {'Longitude': value}

    elif re.match(r'Altitude', string):
        value = convert_to_number(string.replace('\n', ''))
        return {'Height': value}

    elif re.match(r'Data Medicao', string):
        return True

def convert_to_number(string, type=float):
    """take the number from a trailling characters of a string."""
    num = re.search(r'[\-]?[0-9]*\.?[0-9]*$', string).group(0)
    return type(num)

def dump_data(path, data):
    """Storage python object as binary (pickle) file."""

    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_data(path):
    """Load file storaged as binary (pickle)."""

    return pickle.load(open(path, 'rb'))
