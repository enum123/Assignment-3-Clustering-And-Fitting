import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
import sklearn.cluster as cluster


def read(fn):
    """
    Reads a CSV file and returns a pandas DataFrame.
    """
    address = fn
    print(address)
    df = pd.read_csv(address, skiprows=4)
    df = df.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code',
            'Unnamed: 67'])
    return df


def fit_and_forecast(df, Country_name, Ind, tit, tit_fore):
    """
    Funtion to Fit and Predict Using Exponential Funtion.
    """
    # fit exponential growth
    popt, pcorr = opt.curve_fit(exp_growth, df.index, df[Country_name],
                                p0=[4e3, 0.001])
    # much better
    df["pop_exp"] = exp_growth(df.index, *popt)
    plt.figure()
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(df.index, df["pop_exp"], label="fit", color="springgreen")
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.title(tit)
    plt.savefig(Country_name + '.png', dpi=300)
    years = np.linspace(1980, 2030)
    pop_exp = exp_growth(years, *popt)
    sigma = err.error_prop(years, exp_growth, popt, pcorr)
    low = pop_exp - sigma
    up = pop_exp + sigma
    plt.figure()
    plt.title(tit_fore)
    plt.plot(df.index, df[Country_name], label="data")
    plt.plot(years, pop_exp, label="Forecast", color="springgreen")
    # plot error ranges with transparency
    plt.fill_between(years, low, up, alpha=0.5, color="cyan")
    plt.legend(loc="upper left")
    plt.xlabel('Years')
    plt.ylabel(Ind)
    plt.savefig(Country_name + '_forecast.png', dpi=300)
    plt.show()


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1980))
    return f


def get_country_data(df, country_name, start_year, end_year):
    """
    Get Data of Specific Country.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    country_name : TYPE
        DESCRIPTION.
    start_year : TYPE
        DESCRIPTION.
    end_year : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df.T
    df.columns = df.iloc[0]
    df = df.drop(['Country Name'])
    df = df[[country_name]]
    df.index = df.index.astype(int)
    df = df[(df.index > start_year) & (df.index <= end_year)]
    df[country_name] = df[country_name].astype(float)
    return df


def plot_clusters(
        df,
        ind1,
        ind2,
        xlabel,
        ylabel,
        tit,
        n_clu_cen,
        df_fit,
        df_min,
        df_max):
    """
    Clutering the Data based on Features.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    ind1 : TYPE
        DESCRIPTION.
    ind2 : TYPE
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    tit : TYPE
        DESCRIPTION.
    n_clu_cen : TYPE
        DESCRIPTION.
    df_fit : TYPE
        DESCRIPTION.
    df_min : TYPE
        DESCRIPTION.
    df_max : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nc = n_clu_cen  # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc, n_init=10, random_state=0)
    kmeans.fit(df_fit)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure()
    # scatter plot with colours selected using the cluster numbers
    # now using the original dataframe
    scatter = plt.scatter(df[ind1], df[ind2], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(tit)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()


def Elbow_for_clusters(df_fit):
    """
    Find the albow for number of clusters.
    """
    sse1 = []
    for i in range(1, 11):
        kmeans = cluster.KMeans(n_clusters=i, init='k-means++',
                                max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df_fit)
        sse1.append(kmeans.inertia_)
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(range(1, 11), sse1, color='red')
    axs.set_title('Elbow Method')
    axs.set_xlabel('Number of clusters')
    axs.set_ylabel('SSE')


Annual_freshwater_withdrawals_total = read(
    'Annual_freshwater_withdrawals_total.csv')
Urban_population_of_total_population = read(
    'Urban_population _%_of_total_population.csv')
country = 'India'
df_AFW = get_country_data(
    Annual_freshwater_withdrawals_total,
    country,
    1980,
    2020)
df_UP = get_country_data(
    Urban_population_of_total_population,
    country,
    1980,
    2020)

df = pd.merge(df_AFW, df_UP, left_index=True, right_index=True)
df = df.rename(
    columns={
        country +
        "_x": 'Annual_freshwater_withdrawals_total',
        country +
        "_y": 'Urban_population _%_of_total_population'})
df_fit, df_min, df_max = ct.scaler(df)
Elbow_for_clusters(df_fit)

plot_clusters(
    df,
    'Annual_freshwater_withdrawals_total',
    'Urban_population _%_of_total_population',
    'Annual freshwater withdrawals total',
    'Urban Population',
    'Annual freshwater withdrawals vs Urban Population % In India',
    4,
    df_fit,
    df_min,
    df_max)

fit_and_forecast(
    df_AFW,
    'India',
    'Annual Freshwater Withdrawals',
    "Annual Freshwater Withdrawals In India 1980-2020",
    "Annual Freshwater Withdrawals In India Forecast Untill 2030")
fit_and_forecast(
    df_UP,
    'India',
    'Urban Population',
    "Urban Population in India 1980-2020",
    "Urban Population in India Forecast Untill 2030")
