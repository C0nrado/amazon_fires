import pandas as pd
import numpy as np
from collections import Counter
from shapely.geometry import Point
import geopandas as gpd

class Bootstrap():
    """This class provides bootstrap distribution of some statistics provided by a
    *attribute* of some *estimator*."""

    def __init__(self, estimator, attributes, transformation=None, n_samples=1000):
        """*estimator* is a object with a `fit` method to collect the data to be
        processed into a statistics contained in *attribute*. *transformation*
        is a function to act on *attribute* to transform it into the desired
        statistics."""

        self.est_class = estimator
        self.attrs = attributes
        self.n_samples = n_samples
        self.transform = transformation if transformation is not None else self._transform

    def feed(self, *args, alpha=0.05, **kwargs):
        """Takes as *args* amd **kwargs** the same arguments that serves as input to
        the `fit` method in the given *estimator*. *alpha* is the significance level
        to be considered in the confidence interval."""
        
        self.results = []
        self.alpha = alpha

        if 'seed' in kwargs.keys():
            np.random.seed(seed=kwargs.pop('seed'))

        for _ in range(self.n_samples):
            bootstrap_args = self._bootstrap_data(*args, **kwargs)
            est = self.est_class().fit(*bootstrap_args)
            att = getattr(est, self.attrs)
            self.results.append(self.transform(att))
        
        self._process_results()
    
    def _transform(self, arg):
        return arg

    def summary(self):
        interval = '[%.3f --- %.3f]'%(self.alpha/2, 1-self.alpha/2)
        outstring = '%10s%20s%35s\n'%('E[stat]', 'Std.Error', interval)
        outstring += '='*65 + '\n'
        outstring += '%10.3f%20.3f%25.3F%10.3f'%(self.stat, self.se, *self.conf_int)
        return outstring    

    def _bootstrap_data(self, *args, **kwargs):
        """This is a basic function which returns a bootstrap dataset from *args*."""

        n = len(args[0])
        proxies = np.arange(len(args[0]))
        bootstrap_sample_idx = np.random.choice(proxies, n, replace=True)
        return [arg[bootstrap_sample_idx] for arg in args]
    
    def _process_results(self):
        alpha = self.alpha
        self.stat = np.mean(self.results)
        self.se = np.std(self.results)
        self.conf_int = np.quantile(self.results, [alpha/2, 1-alpha/2])


def collect_features(data, feature, **kwargs):
    """Take from each DataFrame in *data* a series indexed by *feature* and stack
    them altogether into a new DataFrame. Accepts same kwargs from pandas concat()."""

    df = pd.concat([station['Data'][feature] for station in data], axis=1, **kwargs)
    return df

def print_frequency(data, title=''):
    """This function prints the number of each instance in *data*."""

    counts = Counter(data).most_common()
    outstring = '%10s%20s\n'%(title, '# of instances')
    outstring += '-'*30 + '\n'
    for k,val in counts:
        outstring += '%10d%20d\n'%(k, val)

    print(outstring)

def extract_locations(data, primary_key=None):
    """Returns a GeoSeries of Point geometries contained in *data*."""
    
    assert all([{'Longitude', 'Latitude'} - set(station.keys()) == set() for station in data])
    points = [(i, Point(station['Longitude'], station['Latitude'])) for i, station in enumerate(data)]
    if primary_key is not None:
        points = map(lambda x: (data[x[0]][primary_key], x[1]), points)
    return gpd.GeoSeries(dict(points))

def extract_fields(data, exclude=()):
    """Return a tuple of all fields in the DataFrames within *data*."""

    features = set()
    for station in data:
        features = features | set(station['Data'].columns.tolist())
    return sorted(list(features - set(exclude)))

def create_df_from_pca(model, data, n_components=None):
    """Returns a DataFrame with columns corresponding to the coefficients from the
    principal component analysis."""

    if n_components is None:
        n_components = model.n_components_
    
    columns = ['coeff_%d'%i for i in range(1, n_components + 1)]
    return pd.DataFrame(model.transform(data)[:,:(n_components)], columns=columns)

def summary_dataset(df):

    def add_line(string, values, line_pattern='{:<15s}{:>20s}\n'):
        for value in values:
            assert isinstance(value, str)
        
        return string + line_pattern.format(*values)
    
    def multiline(string, width=30, sep=' '):
        idx = string.rfind(sep, 0, width)
        out = []
        while len(string) > width:
            out.append(string[:idx])
            string = string[idx+1:]
            idx = string.rfind(sep, 0, width)
        
        out.append(string)
        return len(out), out

    fill_line = lambda x: x + '-'*35 + '\n'
    
    out = ''
    out = fill_line(out)
    out += 'Data Set Summary'.center(35) + '\n'
    out = fill_line(out)

    if isinstance(df.index, pd.PeriodIndex) or isinstance(df.index, pd.DatetimeIndex):
        out = add_line(out, ('Starts at', str(df.index.min())))
        out = add_line(out, ('Ends at', str(df.index.max())))
        out = add_line(out, ('Type', str(df.index.dtype)))
    
    out = add_line(out, ('Entries', str(len(df))))
    
    n_lines, fields_lines = multiline(', '.join(df.columns), 20)
    
    for values in zip(['Fields'+' ('+str(len(df.columns))+')']+['']*(n_lines-1), fields_lines):
        out = add_line(out, values)
    
    out = add_line(out, ('NULL ', str(df.isnull().sum().sum())))
    out = add_line(out, ('Memory[KB]', str(round(df.memory_usage().sum()/(2**10),1))))
    
    out = fill_line(out)
    out += 'Data Types'.center(35) + '\n'
    out = fill_line(out)

    for values in (tuple(d.split()) for d in df.dtypes.to_string().split('\n')):
        out = add_line(out, values)
    
    # Statistics columh
    
    col2_body = df.describe().round(1).to_string()
    line_width = len(col2_body.split('\n')[0])
    
    col2 = '' 
    col2 = fill_line(col2)
    col2 += 'Statistics'.center(line_width) + '\n'
    col2 = fill_line(col2)
    col2 += col2_body
    out = out.split('\n')
    col2 = col2.split('\n')


    final = [line1 + '|'.center(5) + line2
                    for line1, line2 in zip(out, col2 + ['']*(len(out) - len(col2) -1))]
    final_width = len(final[0])
    
    out = '='*(final_width) + '\n'
    out += '\n'.join(final) + '\n'
    out += '-'*final_width + '\n'
    out += '='*(final_width)

    return out

def get_peak_month(df, field):
    """Takes a time series indexed pandas DataFrame and return the month when
    **field** hat its peak value."""

    series = df.assign(Year = lambda x: x.index.year,
                     Month = lambda x: x.index.month) \
             .pivot_table(index='Year', columns='Month', values = field) \
             .apply(lambda s: np.nanargmax(list(s)) + 1, axis=1)

    return pd.Series(series.values, index=pd.PeriodIndex(series.index, freq='A'))
