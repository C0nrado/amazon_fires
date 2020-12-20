import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import re
import scipy.stats as stats
from statsmodels.tsa.stattools import acf
from matplotlib.ticker import AutoMinorLocator

def streamplot(data, interval=None, marker=None, ax=None, **kwargs):
    """Plots multiple streams (sequences) of a variable collected independently.
    The *data* is a DataFrame with columns as a series, or a list of series entirely
    where each series portrays a particular stream of the variable measured.
    *Interval* accepted as input `[n]sigma` or `robust`."""

    ax_kwargs = {'color':'C0', 'alpha':.4}
    # ax_kwargs.update(**kwargs)
   
    if isinstance(data, list):
        data = pd.concat(data, axis=1)
    xmin, xmax = data.index[0], data.index[-1]
 
    if ax is None:
        _, ax = plt.subplots()
    
    if interval is not None:

        interval = _parse_interval(interval)
        central_line = data.mean(axis=1)
        central_line.plot(ax=ax, color='k', lw=1.2, marker=marker, **kwargs)
        # ax.plot(data.index, central_line, color='k', lw=1.2, marker=marker)

        if interval[1].lower() == 'sigma':
            n = 1 if interval[0] is None else int(interval[0])
            sigma = data.std(axis=1, ddof=1)
            upper = central_line + sigma * n
            lower = central_line - sigma * n
        elif interval[1].lower() == 'robust':
            upper = data.quantile(.95, axis=1)
            lower = data.quantile(.05, axis=1)
        
        ax.plot(data.index, upper, color='none')
        ax.plot(data.index, lower, color='none')
        ax.fill_between(data.index, upper, lower, **ax_kwargs)
    
    else:
        data.plot(ax=ax, lw=.8, **kwargs)
    
    ax.set_xlim(xmin, xmax)
    ax.grid(color='gray', lw=.5)
    ax.tick_params(axis='both', which='both', direction='in', length=6, top=True, right=True)
    ax.tick_params(axis='x', which='major', length=10)

def parallelplot(df, category, centroids=False, interval=False, color=None, alpha=None, ax=None):
    """Plot records of DataFrame *df* in parallel coordinates in accordance to the categoriacal field *category*."""

    fields = df.drop(category, axis=1).columns
    cat_values = df[category].unique()
    xs = np.arange(len(fields))
    ymin, ymax = df.min().min() * 1.1, df.max().max() * 1.1
    lineHandle = []
 
    if ax is None:
        _, ax = plt.subplots()
    ax.set_ylim(ymin, ymax)

    if centroids:
        data = df.groupby(category).mean().reindex(cat_values).to_numpy()
        if interval == 'std':
            std = df.groupby(category).std().reindex(cat_values)
            sup = data + std.to_numpy() * 1.96
            low = data - std.to_numpy() * 1.96
        if interval == 'robust':
            pctl = df.groupby(category).apply(lambda s: s.quantile([.05, .95])).reindex(cat_values, level=0).swaplevel()
            sup = pctl.loc[0.95, fields].to_numpy()
            low = pctl.loc[0.05, fields].to_numpy()

        for i, ys in enumerate(data):
            line = ax.plot(xs, ys, color='k', lw=1.5, marker='s', mfc='C'+str(i), ms=7)
            lineHandle.append(line[0])
            if interval:
                ax.fill_between(xs, sup[i], low[i], alpha=alpha, lw=.5, color='C'+str(i))

    else:
        data = [df[df[category] == value][fields].to_numpy() for value in cat_values]
        for i, ys in enumerate(data):
            lines = ax.plot(ys.T, color='C'+str(i), alpha=alpha)
            lineHandle.append(lines[0])
    
    for i in xs[1:-1]:
        ax.vlines(i, ymin, ymax, lw=.5)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_xticklabels(fields)
    ax.set_ylim(ymin, ymax)
    ax.legend(lineHandle, cat_values, title='Cluster')

    return ax

def _parse_interval(interval):
    """this function parses interval string inputs."""

    if interval is None:
        return None
    else:
        match = re.match(r'([1-3])?(sigma|robust)', interval)
        try:
            match.groups()
        except:
            raise ValueError('Only [n]sigma or robust are valid inputs.')
    return match.groups()

def plot_nullvalues(data, ax=None):
    """Plots a matrix for # of null on a grid of features x wheather station."""

    df = pd.concat([station['Data'].isna().sum(axis=0).rename(station['Code']) for station in data], axis=1)
    features = df.index
    codes = df.columns

    if ax is not None:
        plt.sca(ax)

    plt.imshow(df, cmap='Blues', aspect='auto')
    plt.xticks(range(len(codes)), labels=codes, rotation=45, ha='right')
    plt.yticks(range(len(features)), labels=features)
    plt.title('Total occurrences of NULL values')
    plt.xlabel('Station Code') 
    plt.colorbar()

def decompositionplot(obj, trend=True, seasonal=True, residual=True, ax=None, title=None):
    """Plots optionally the `trend`, `seasonal`, and/or `residual` of an STL object or
    other with this same attributes containing sequences or arrays."""

    def att_gen(trend, seasonal, residual):
        for att, value in zip(['trend', 'seasonal', 'resid'], [trend, seasonal, residual]):
            if value:
                yield att

    plot_props = {'trend':
        # Trend props
        dict(lw=1.2, color='C0'),
        'seasonal':
        # Seasonal props
        dict(lw=.5, color='k')    }

    if isinstance(obj.observed, pd.Series):
       obj = pd.DataFrame({att: getattr(obj, att) for att in att_gen(trend, seasonal, residual)}) 

    ncols = (trend or seasonal) * 2 + residual
    if ax is None:
        fig = plt.figure(figsize=(8*(trend or seasonal) + 4 * residual, 4), constrained_layout=True)
        gs = gridspec.GridSpec(1, ncols, figure=fig)
        ax1 = gs[0: 1+ (trend or seasonal)]
        ax2 = gs[-1]
        ax = [ax1, ax2]

        if title is not None:
            fig.suptitle(title)
    else:
        assert len(ax) == ncols, "! axes for Trend/Seasonal plot and 1 axes for Residual plot."
        ax1, ax2 = ax

    if trend or seasonal:
        ax1 = fig.add_subplot(ax1)
        ax1.set_facecolor('#E5E8ED')
        for att in att_gen(trend, seasonal, False):
            getattr(obj, att).plot(ax=ax1, label=att.capitalize(), **plot_props[att])

        ax1.set_title('Seasonal Trend Decomposition')
        ax1.grid(lw=1.5, color='w', alpha=.7)
        ax1.legend()
        # ax1.set_xlim(xmin, xmax)

    if residual:
        ax2 = fig.add_subplot(ax2)
        ax2.set_facecolor('#E5E8ED')
        getattr(obj, 'resid').hist(bins=15, edgecolor='w', lw=.5,zorder=2)
        ax2.set_yticklabels('')
        ax2.set_title('Residuals')
        ax2.grid(lw=1.5, color='w', alpha=.7)
        ax2.tick_params(left=False)
    
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    return fig.axes

def boxplot(data, title=None, ax=None, custom_func='default', **kwargs):
    """Wrapper around matplotlib boxplot helper function with a optional argument
    for a callable for plot customization."""

    if ax is None:
        _, ax = plt.subplots()
    
    ax.set_facecolor('#E5E8ED')
    ax.grid(axis='both', lw=1.5, color='w', alpha=.75)
    boxplot = ax.boxplot(data, **kwargs)
    ax.set_title(title)

    if custom_func == 'default':
        custom_func = _custom_boxplot
        custom_func(boxplot)
 
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax

def _custom_boxplot(plot):
    for box in plot['boxes']:
        box.set_lw(1.5)
        box.set_edgecolor('w')
        box.set_color('C0')

    for whisker in plot['whiskers']:
        whisker.set_linewidth(2)
        whisker.set_color('C0')

    for cap in plot['caps']:
        cap.set_color('C0')
        cap.set_linewidth(2)

    for flier in plot['fliers']:
        flier.set_marker('d')
        flier.set_markerfacecolor('k')
        flier.set_mec('k')
    
    for median in plot['medians']:
        median.set_linewidth(1.5)
        median.set_color('lightblue')

def scatterplot(df, x, y, col=None, legend=False, colorbar=False, legend_kwds=None, cbar_kwds=None, ax=None, **kwargs):
    """This plotting function is streamlined for plotting on DataFrames."""

    kw = dict(cmap='tab10')
    kw.update(**kwargs)
    kw['c'] = getattr(df, kwargs['c']) if kwargs.get('c') else None

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if col in df.columns:
        for i, value in enumerate(df[col].unique()):
            _df = df.query("%s == @value"%col)
            
            if legend:
                kw.update({'label': value})

            ax.scatter(x=getattr(_df, x), y=getattr(_df, y), color=plt.get_cmap(kw['cmap'])(i), **kw)

        ax.update_datalim(df[[x, y]].values)
    
    else:
        plot = ax.scatter(x=getattr(df, x), y=getattr(df, y), **kw)
        
        if colorbar:
            if cbar_kwds is None:
                cbar_kwds = {}
            ax.figure.colorbar(plot, **cbar_kwds)

    if legend:
        if legend_kwds is None:
            legend_kwds = {}
    
        ax.legend(**legend_kwds)
    
    return ax

def autocorrplot (series, nlags='auto', autocorr_fn=None, ax=None, **kwargs):
    """Plot the autocorrelation coefficients of a time series. *autocorr_fn* must
    has *nlags* as parameter for number of lags to be computed. Default is same
    as statsmodels."""

    plot_params = {'title': "Autocorrelation Plot",
              'alphas': [0.05, 0.01]}
    plot_params.update(**kwargs)
    
    if autocorr_fn is None:
        autocorr_fn = acf
    autocoeff = series.pipe(autocorr_fn, nlags=nlags)

    assert nlags == 'auto' or (isinstance(nlags, int) and nlags > 0), "If not `auto` it must be a positive integer."

    if nlags == 'auto':
        nlags = len(autocoeff) + 1
    if ax is None:
        _, ax = plt.subplots()

    ax.vlines(np.arange(nlags), 0, autocoeff)
    x_range = ax.get_xlim()

    for alpha, ls in zip(plot_params['alphas'], ['solid', (0, (9, 9))]):
        ax.hlines(stats.norm.ppf([alpha/2, 1-alpha/2]) * 1/len(series)**.5, *x_range, ls=ls, lw=.8, color='red')

    ax.set_xlim(*x_range)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(lw=.5, color='gray')
    ax.tick_params(axis='both', which='both', direction='in', length=6, top=True, right=True)
    ax.tick_params(axis='x', which='major', length=10)

    ax.set_xlabel('Lags')
    ax.set_title(plot_params['title'])

    return ax