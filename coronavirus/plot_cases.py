"""
# python3 unicode
# author:Steven Huang 25/04/20
# function: Query cases of COVID-19 from website
# World case statistics by time
"""

from cmath import nan
import os
import random
import datetime
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from predict_cases import plotDataAx, binaryDf
from predict_cases import gSaveBasePath, gSaveChangeData
from predict_cases import gSaveCountryData, gSavePredict
from json_update import get_datetime

from common_path import create_path, traverse_files, get_file_names
from progress_bar import SimpleProgressBar

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 150 #high resolution 100~300

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('font', family='Times New Roman')
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=SMALL_SIZE)   # fontsize of the figure title


def init_folder():
    directory = os.path.dirname(os.path.realpath(__file__))
    # print(directory)
    create_path(os.path.join(directory, gSaveBasePath))
    create_path(os.path.join(directory, gSaveChangeData))
    create_path(os.path.join(directory, gSaveCountryData))


def get_datetime_str():
    return str(' Date:') + get_datetime()


def plotCountriesFromOurWorld(csvpath=r'./OurWorld/'):  # From ourworld data
    saveCountriesInfoFromCsv(csvpath)

    casesFile = os.path.join(csvpath, 'cases.csv')
    df = read_csv(casesFile)
    plotWorldCases(df)

    countriesPath = r'./dataCountry/'
    dfToday = getAllOurWorldNew(countriesPath)
    dfToday.to_csv(r'./OurWorld/today.csv',index=True)

    plotContinentCases(dfToday)
    plotCountriesTopCases(dfToday)
    plotContinentCountriesCases(dfToday)
    plotCountryCases(dfToday, countriesPath)


def plotData(df, title='', kind='line', y=None, fName='', logy=False,
             save=True, show=True, left=0.1, bottom=0.14, top=0.92,
             color='#1f77b4', grid=False, figsize=None):
    # fontsize = 4
    if figsize:
        ax = df.plot(kind=kind, title=title, y=y, color=color, grid=grid,
                     logy=logy, figsize=figsize, xlabel='', ylabel='')
    else:
        ax = df.plot(kind=kind, title=title, y=y, color=color, grid=grid,
                     logy=logy, xlabel='', ylabel='')

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # plt.setp(ax.get_yticklabels()) #fontsize=fontsize
    # plt.tight_layout()
    plt.subplots_adjust(left=left, bottom=bottom, right=0.96, top=top, wspace=None, hspace=None)
    if save:
        plt.savefig(fName)
    if show:
        plt.show()


def read_csv(file):
    df = pd.read_csv(file)
    # print(df.describe().transpose())
    # print(df.head())
    # df.set_index(["Location"], inplace=True)
    # print('df.columns=',df.columns)
    # print('df.dtypes = ',df.dtypes)
    # df = df.apply(pd.to_numeric, axis=0)
    # print('df.dtypes = ',df.dtypes)
    return df


def getAllOurWorldNew(csvpath=r'./dataCountry/'):
    def getCountryNewestLine(file):
        # country = get_file_names(file)[0]
        # print('country=', country)

        df = read_csv(file)
        # df = df.drop(columns=['continent','iso_code'])
        # df = df.dropna(subset=['total_vaccinations'])
        df.set_index(["date"], inplace=True)
        df = df.sort_values(by=['date'], ascending=True)

        if not df.empty:
            # print('df=', df.head())
            if df.shape[0]>1:
                newestLine = df.iloc[[-1]]
                # print("newestLine=\n", newestLine);
                if np.isnan(newestLine['total_cases'][0]): # filter the last line not equal to 0
                    if df.shape[0]>2:
                        newestLine = df.iloc[[-2]]
                        return newestLine
                return newestLine
            # print("newestLine=\n", newestLine);
        return None

    dfAll = []
    continents = ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania', 'World']
    for fileCountry in traverse_files(csvpath, 'csv'):
        # fileCountry = os.path.join(vaccPath, 'vaccination_Burkina Faso.csv')
        line = getCountryNewestLine(fileCountry)
        # print('line=', line, type(line))

        continent = get_file_names(fileCountry)[0]
        # print('fileCountry=', fileCountry, continent)

        # if continent in continents:
            # print('continents=', fileCountry, line['total_cases'][0])

        if line is not None:
            dfAll.append(line)
        # break

    dfAll = pd.concat(dfAll)
    # print("dfAll.head=\n", dfAll.head())
    # print("dfAll.columns=\n",dfAll.columns, dfAll.shape)
    dfAll.set_index(["location"], inplace=True)
    return dfAll


def plotCountryCases(dfToday, path):
    def getTopDataCountries(df, top, columnLabel, ascend=False):
        df = df.sort_values(by=[columnLabel], ascending=ascend)
        dfData = df.iloc[:top, :]  # [df.columns.get_loc(columnLabel)]
        return dfData  # dfData['location']

    def getDataCountries(df, countriesList):
        # dfData = df.loc[df['location'].isin[countriesList]]
        dfData = df.filter(items=countriesList, axis=0)
        # dfData = df.filter(like = 'aus', axis=0)
        return dfData

    df = dfToday[dfToday['continent'].notnull()]  # remain countries not continent
    # print('df=', df)

    top = 15
    days = 60
    plotAll = []

    columnLabel = 'total_cases'
    dfData = getTopDataCountries(df, top, columnLabel)
    title = 'Top ' + str(top) + ' Confirmed' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_Confirmed.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    columnLabel = 'total_deaths'
    dfData = getTopDataCountries(df, top, columnLabel)
    title = 'Top ' + str(top) + ' Deaths' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_Deaths.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    columnLabel = 'new_deaths'
    dfData = getTopDataCountries(df, top, columnLabel)
    title = 'Top ' + str(top) + ' New Deaths' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_NewDeaths.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    columnLabel = 'new_cases'
    dfData = getTopDataCountries(df, top, columnLabel)
    title = 'Top ' + str(top) + ' New Cases' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_NewConfirmed.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    dfContinent = dfToday[dfToday['continent'].isnull()]  # remain  continent
    oters = ['OWID_HIC', 'OWID_LIC', 'OWID_LMC', 'OWID_UMC']
    dfContinent = dfContinent.loc[~dfContinent['iso_code'].isin(oters)]

    dfContinent = dfContinent.drop(index=['World'])
    print('dfContinent=', dfContinent)

    columnLabel = 'new_deaths'
    dfData = getTopDataCountries(dfContinent, -1, columnLabel)
    title = 'Continent New Deaths' + get_datetime_str()
    fileName = gSaveBasePath + 'continent_NewDeaths.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    columnLabel = 'new_cases'
    dfData = getTopDataCountries(dfContinent, -1, columnLabel)
    title = 'Continent New Cases' + get_datetime_str()
    fileName = gSaveBasePath + 'continent_NewConfirmed.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    columnLabel = 'mortality'
    days = 0  # plot all dates
    '''
    dfData = getTopDataCountries(df, top, columnLabel)
    title = 'Top ' + str(top) + ' mortality' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_Mortality.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    dfData = getTopDataCountries(df, top, columnLabel, ascend=True)
    title = 'Top ' + str(top) + ' lowest mortality' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_MortalityLow.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))
    '''

    countries = ['Italy', 'Vietnam', 'World', 'Germany', 'United Kingdom', 'United States',
                 'South Korea', 'Japan', 'New Zealand']
    dfData = getDataCountries(df, countries)
    title = 'Typical countries' + ' mortality' + get_datetime_str()
    fileName = gSaveBasePath + 'countries_MortalityTC.png'
    plotAll.append((columnLabel, dfData, title, fileName, days))

    for i, value in enumerate(plotAll):
        columnLabel, dfData, title, fileName, days = value
        # print('plotConuntryCasesByTime=', i, columnLabel, dfData.shape, title, fileName, days)
        plotConuntryCasesByTime(path, dfData, columnLabel, title, fileName, days)


def plotCountryCasesStyle1(dfAll, coloumnLabel, title, fileName, days=60):
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    print('title=', title)
    ax.set_title(title)
    # plt.title(title)
    for country, df in dfAll:
        df.set_index(["date"], inplace=True)
        df.sort_index(inplace=True)

        # y = df[coloumnLabel]
        # #y = InterpolationDf(dateIndex, y)
        # inter = 12
        # y = y[::inter]

        y = df.iloc[-1 * days:, [df.columns.get_loc(coloumnLabel)]]  # recent 30 days
        # plotCountryAx(ax,y.index, y, label=country)
        plotDataAx(ax, y.index, y[coloumnLabel], label=country, fontsize=MEDIUM_SIZE)
        # break

    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()


def plotCountryCasesStyle2(dfAll, coloumnLabel, title, fileName, days=120):
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    # print('title=', title)
    ax.set_title(title)
    # plt.title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = [cm.jet(float(i) / len(dfAll)) for i in range(len(dfAll))]
    random.shuffle(colors)

    for i, (country, df) in enumerate(dfAll):
        df.set_index(["date"], inplace=True)
        df.sort_index(inplace=True)

        # y = df[coloumnLabel]
        # #y = InterpolationDf(dateIndex, y)
        # inter = 12
        # y = y[::inter]

        y = df.iloc[-1 * days:, [df.columns.get_loc(coloumnLabel)]]  # recent 30 days

        color = colors[i]

        # print('type(y), type(xIndex)=', type(y), y.shape, y.index, type(y.index))
        # plotCountryAx(ax,y.index, y, label=country)
        plotDataAx(ax, y.index, y, country, fontsize=MEDIUM_SIZE, color=color)
        ax.text(y.index[-1], y.iloc[-1, [0]], country, color=color)
        # break

    bottom, top = plt.ylim()
    bottom = max(0, bottom)

    inter = (top - bottom) // 25
    for i in range(int(bottom), int(top), int(inter)):
        plt.plot(y.index, [i] * len(y.index), "--", lw=0.5, color="black", alpha=0.3)

    plt.legend(loc='upper left')
    # plt.yscale('log')
    # plt.ylim(0)
    plt.tight_layout()
    plt.savefig(fileName)
    plt.show()


def plotConuntryCasesByTime(path, dfCountries, coloumnLabel, title, fileName, days=60):
    # dateIndex = getDateIndex()
    # print('dfCountries=\n', dfCountries)
    # countries = list(dfCountries.location.unique())
    countries = list(dfCountries.index.unique())
    print('countries=', countries)
    dfAll = []
    for country in countries:
        name = country + '.csv'
        # print(name)
        file = os.path.join(path, name)
        df = read_csv(file)
        df = df.fillna(0)
        df = df.sort_values(by=['date'], ascending=True)
        # print('name=',name,'df=', df)
        dfAll.append((country, df))

    plotCountryCasesStyle1(dfAll, coloumnLabel, title, fileName, days=days)
    # plotCountryCasesStyle2(dfAll, coloumnLabel, title, fileName, days=days)


def getTopData(df, top, columnLabel, binary=True, ascend=False):
    df = df.sort_values(by=[columnLabel], ascending=ascend)
    dfData = df.iloc[:top, [df.columns.get_loc(columnLabel)]]

    if binary:
        dfData = binaryDf(dfData, labelAdd=True)
    return dfData


def plotCountriesTopCases(df):
    # df.set_index(["location"], inplace=True)

    dfWorld = df.loc['World']
    # dfWorld.fillna(0, inplace=True)  # Pandas SettingWithCopyWarning
    dfWorld = dfWorld.fillna(0)

    df = df[df['continent'].notnull()]

    # print(df.head())
    # print('df.columns=', df.columns)
    # print('dfWorld=', dfWorld)

    # total_cases new_cases total_deaths new_deaths
    top = 50
    kind = 'barh'
    plotAll = []

    columnLabel = 'total_cases'
    dfData = getTopData(df, top, columnLabel)

    title = 'World top ' + str(top) + ' Confirmed(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    color = '#1f77b4'
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'new_cases'
    dfData = getTopData(df, top, columnLabel)
    title = 'World top ' + str(top) + ' NewCase(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'total_cases_per_million'
    dfData = getTopData(df, top, columnLabel)
    title = 'World top ' + str(top) + ' Cases per Million(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'total_deaths'
    dfData = getTopData(df, top, columnLabel)
    title = 'World top ' + str(top) + ' Deaths(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    color = 'r'
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'new_deaths'
    dfData = getTopData(df, top, columnLabel)
    title = 'World top ' + str(top) + ' New Deaths(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'mortality'
    dfData = getTopData(df, top, columnLabel)
    title = 'World top ' + str(top) + ' Mortality(World: ' + str(dfWorld[columnLabel]) + ')' + get_datetime_str()
    plotAll.append((columnLabel, dfData, kind, title, color))

    columnLabel = 'mortality'
    dfData = getTopData(df, top, columnLabel, ascend=True)
    title = 'World lowest ' + str(top) + ' Mortality(World: ' + str(dfWorld[columnLabel]) + ')' + get_datetime_str()
    plotAll.append((columnLabel, dfData, kind, title, color))

    for i, value in enumerate(plotAll):
        columnLabel, dfData, kind, title, color = value
        fileName = os.path.join(gSaveBasePath, str(i + 1) + '.png')
        plotData(dfData, title=title, kind=kind, y=[columnLabel], fName=fileName,
                 save=True, show=False, color=color, figsize=(8, 5), left=0.3, bottom=0.1)
    plt.show()


def plotContinentCountriesCases(df):
    # df.set_index(["location"], inplace=True)

    dfWorld = df.loc['World']
    dfWorld = dfWorld.fillna(0)
    df = df[df['continent'].notnull()]

    continents = list(df.continent.unique())
    print('continents=', continents, len(continents))

    top = 10
    kind = 'bar'  # ['line','bar','barh','hist','box','kde','density','area']
    plotAll = []

    columnLabel = 'new_cases'
    for continent in continents:
        # print('continent=', continent) #['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania']
        dfCountries = df[df['continent'] == continent]

        color = 'r'
        dfData = getTopData(dfCountries, top, columnLabel, binary=False)
        # title = continent + ' top ' + str(top) + ' NewCase(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
        title = continent + ' top ' + str(top) + ' NewCase'
        plotAll.append((columnLabel, dfData, kind, title, color))

    print('plotAll len=', len(plotAll))

    nrow = 2
    ncol = 3
    title = 'Continent today\'s new cases(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 6))
    fig.suptitle(title)  # fontsize=10
    for i, value in enumerate(plotAll):
        columnLabel, dfData, kind, title, color = value

        ax = dfData.plot(ax=axes[i // ncol, i % ncol], kind=kind, title=title, y=[columnLabel],
                         color=color, grid=False, logy=False, legend=False, xlabel='', ylabel='')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.tight_layout()

    fileName = gSaveBasePath + 'continentTopCountries_NewCases' + '.png'
    plt.savefig(fileName)

    plotAll = []
    columnLabel = 'new_deaths'
    for continent in continents:
        dfCountries = df[df['continent'] == continent]
        color = 'r'

        dfData = getTopData(dfCountries, top, columnLabel, binary=False)
        # title = continent + ' top ' + str(top) + ' NewCase(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
        title = continent + ' top ' + str(top) + ' New Deaths'
        plotAll.append((columnLabel, dfData, kind, title, color))

    title = 'Continent today\'s new deaths(World: ' + str(int(dfWorld[columnLabel])) + ')' + get_datetime_str()
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 6))
    fig.suptitle(title, fontsize=10)
    for i, value in enumerate(plotAll):
        columnLabel, dfData, kind, title, color = value

        ax = dfData.plot(ax=axes[i // ncol, i % ncol], kind=kind, title=title, y=[columnLabel],
                         color=color, grid=False, logy=False, legend=False, xlabel='', ylabel='')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.tight_layout()

    fileName = gSaveBasePath + 'continentTopCountries_NewDeaths' + '.png'
    plt.savefig(fileName)
    plt.show()


def plotWorldCases(df):
    df = df[df['location'] == 'World']
    df = df.drop(columns=['location', 'continent', 'iso_code'])
    df = df.dropna(subset=['total_cases'])

    print('columns=', df.columns)

    df.set_index(["date"], inplace=True)
    print(df.head())

    df = df.iloc[::5]  # drop some lines
    # dfWorld = binaryDf(dfWorld,False) #drop half
    print('head after drop:\n', df.head())

    # kinds = ['line','bar','barh','hist','box','kde','density','area']

    title = 'World Cases,' + get_datetime_str()
    y = ['total_cases']
    fileName = gSaveBasePath + 'World_Cases.png'
    plotData(df, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5))

    title = 'World New Cases,' + get_datetime_str()
    y = ['new_cases']
    fileName = gSaveBasePath + 'World_NewCases.png'
    plotData(df, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5))

    title = 'World Deaths,' + get_datetime_str()
    y = ['total_deaths']
    fileName = gSaveBasePath + 'World_Deaths.png'
    plotData(df, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5), color='r')

    title = 'World New Deaths,' + get_datetime_str()
    y = ['new_deaths']
    fileName = gSaveBasePath + 'World_NewDeaths.png'
    plotData(df, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5), color='r')

    title = 'World Mortality,' + get_datetime_str()
    y = ['mortality']
    fileName = gSaveBasePath + 'World_Mortality.png'
    plotData(df, title=title, kind='line', y=y, fName=fileName, show=False, figsize=(8, 5), color='r', grid=True)

    newRecentDays = 60
    dfWorldNew = df.iloc[-1 - newRecentDays:-1, :]
    strRecent = '{} days'.format(newRecentDays)

    title = 'World Recent {} New Cases,'.format(strRecent) + get_datetime_str()
    y = ['new_cases']
    fileName = gSaveBasePath + 'World_RecentNewCases.png'
    plotData(dfWorldNew, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5))

    title = 'World Recent {} New Deaths,'.format(strRecent) + get_datetime_str()
    y = ['new_deaths']
    fileName = gSaveBasePath + 'World_RecentNewDeaths.png'
    plotData(dfWorldNew, title=title, kind='bar', y=y, fName=fileName, show=False, figsize=(8, 5), color='r')
    plt.show()


def plotContinentCases(df):
    def plotContienet(dfContinent, y, title, fileName, color):
        if 0:
            plotData(dfContinent, title=title, kind='bar', y=y, fName=fileName, color=color, bottom=0.17, top=0.9)
        else:
            # dfContinentTable = dfContinent.iloc[:, [2,3,5,6]]
            dfContinentTable = dfContinent.iloc[:, [dfContinent.columns.get_loc(i) for i in y]]
            dfContinentTable = dfContinentTable.astype('int32')

            # start to plot bar and table
            # y1 = dfContinent['total_cases']#dfContinent.loc[:,['total_cases', 'total_deaths']]
            # y2 = dfContinent['total_deaths']
            # plt.bar(y1.index, y1, width=0.4) #bottom=0, color=color
            # plt.bar(y2.index, y2, width=0.4) #bottom=0, color=color
            ax = dfContinent.plot(kind='barh', title=title, y=y, color=color,
                                  figsize=(8, 5), ylabel='', xlabel='', fontsize=10)  # grid=True, logy=True
            # ax.get_xaxis().set_visible(False)
            ax.get_xaxis().tick_top()
            the_table = plt.table(cellText=dfContinentTable.values,
                      rowLabels=dfContinentTable.index,
                      colLabels=dfContinentTable.columns,
                      loc='bottom')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            # the_table.scale(2, 2)

            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            # plt.setp(ax.get_yticklabels()) #fontsize=fontsize

            plt.subplots_adjust(left=0.15, bottom=0.25, right=None, top=0.82, wspace=None, hspace=None)
            plt.tight_layout()
            plt.savefig(fileName)
            plt.show()
            # print('dfContinentTable=\n', dfContinentTable)

    dfContinent = df[df['continent'].isnull()]
    oters = ['OWID_HIC', 'OWID_LIC', 'OWID_LMC', 'OWID_UMC']
    dfContinent = dfContinent.loc[~dfContinent['iso_code'].isin(oters)]

    # dfContinent.set_index(["location"], inplace=True)
    dfContinent = dfContinent.fillna(0)
    print('dfContinent=\n', dfContinent)

    worldTotalCases = int(dfContinent.loc['World']['total_cases'])
    worldTotalDeaths = int(dfContinent.loc['World']['total_deaths'])
    worldNewCases = int(dfContinent.loc['World']['new_cases'])
    worldNewDeaths = int(dfContinent.loc['World']['new_deaths'])

    # print('world=', worldTotalCases, worldTotalDeaths) #
    dfContinent = dfContinent.drop(index=['International', 'World'])

    # print('dfContinent=\n', dfContinent)
    # print('dfContinent.columns=\n', dfContinent.columns)

    wroldstr = 'World total cases:{}, total deaths:{}'.format(worldTotalCases, worldTotalDeaths)
    title = wroldstr + '\n' + get_datetime_str()
    y = ['total_cases', 'total_deaths']
    fileName = gSaveBasePath + 'World_casesContinent.png'
    color = ['#1f77b4', 'red']
    dfContinent = dfContinent.sort_values(by=[y[0]], ascending=False)
    plotContienet(dfContinent, y, title, fileName, color)

    wroldstr = 'World new cases:{}, new deaths:{}'.format(worldNewCases, worldNewDeaths)
    title = wroldstr + '\n' + get_datetime_str()
    y = ['new_cases', 'new_deaths']
    fileName = gSaveBasePath + 'World_newCasesContinent.png'
    dfContinent = dfContinent.sort_values(by=[y[0]], ascending=False)
    plotContienet(dfContinent, y, title, fileName, color)


def getCountryNewCasesAndDeathsDf(pdDate):
    pdDate['NewConfirmed'] = 0
    pdDate['NewDeaths'] = 0
    # print(pdDate.head(5))
    for i in range(pdDate.shape[0] - 1):
        newConfirmed = pdDate['Confirmed'].iloc[i + 1] - pdDate['Confirmed'].iloc[i]
        newConfirmed = max(newConfirmed, 0)
        pdDate.iloc[i + 1, pdDate.columns.get_loc("NewConfirmed")] = newConfirmed

        newDeaths = pdDate['Deaths'].iloc[i + 1] - pdDate['Deaths'].iloc[i]
        newDeaths = max(newDeaths, 0)
        pdDate.iloc[i + 1, pdDate.columns.get_loc("NewDeaths")] = newDeaths

        # pdDate['NewConfirmed'].iloc[i+1] = pdDate['Confirmed'].iloc[i+1] - pdDate['Confirmed'].iloc[i]
        # pdDate['NewDeaths'].iloc[i+1] = pdDate['Deaths'].iloc[i+1] - pdDate['Deaths'].iloc[i]

    # print(pdDate.head(5))
    return pdDate


def saveCountriesInfo(all):
    countries = all[-1][1]['Location']
    bar = SimpleProgressBar(total=len(countries), title='Save Country Files', width=30)
    for k, i in enumerate(countries):
        df = getCountryDayData(i, all)
        # print(df.head(5))
        df = getCountryNewCasesAndDeathsDf(df)
        df.to_csv(gSaveCountryData + i + '.csv', index=True)
        bar.update(k + 1)


def saveAllCases2CSV(file, casesFile):
    df = read_csv(file)
    print(df.head())

    # df = df[df['continent'].notnull()]
    # df = df.drop(columns=['continent','iso_code'])
    print('df.columns=', df.columns)
    nColumns = ['location', 'continent', 'iso_code', 'date', 'total_cases', 'new_cases',
                'new_cases_smoothed', 'total_deaths', 'new_deaths',
                'new_deaths_smoothed', 'total_cases_per_million',
                'new_cases_per_million', 'new_cases_smoothed_per_million',
                'total_deaths_per_million', 'new_deaths_per_million',
                'new_deaths_smoothed_per_million']

    dfCases = df.loc[:, nColumns]
    dfCases['mortality'] = round(dfCases['total_deaths'] / dfCases['total_cases'], 6)  # round 6

    dfCases.set_index(["location"], inplace=True)
    print(dfCases.head())
    dfCases.to_csv(casesFile, index=True)


def saveCountriesInfoFromCsv(csvPath):
    file = os.path.join(csvPath, 'owid-covid-data.csv')
    casesFile = os.path.join(csvPath, 'cases.csv')
    saveAllCases2CSV(file, casesFile)
    df = read_csv(casesFile)

    create_path(gSaveCountryData)
    countries = list(df.location.unique())
    # print('countries=', countries, len(countries))
    bar = SimpleProgressBar(total=len(countries), title='Save Country Files', width=30)
    for i, country in enumerate(countries):
        name = country + '.csv'
        # print(name)
        file = os.path.join(gSaveCountryData, name)
        # print(df)
        # print(file)
        dfCountry = df[df['location'] == country]
        # dfCountry = dfCountry.drop(columns=['location'])
        dfCountry.set_index(["date"], inplace=True)
        # print(dfCountry)
        dfCountry.to_csv(file, index=True)

        bar.update(i + 1)
        # break


def plotCountryInfo(all, column='Confirmed'):
    days = 30
    countriesNumbers = 15

    # countries=['United States','Spain','Italy']
    countries = all[-1][1]['Location'][1:countriesNumbers]
    # print(countries)

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)

    for i in countries:
        df = getCountryDayData(i, all)
        df = getCountryNewCasesAndDeathsDf(df)
        # print(df.head(5))
        # df = binaryDf(df,labelAdd=False)
        df = df.iloc[-1 * days:, :]  # recent 30 days
        title = column + ' Cases, Top ' + str(countriesNumbers) + ' countries, ' + 'Recent ' + str(days) + ' days'
        plotCountryAx(ax, df['Date'], df[column], label=i, title=title)

    if column not in ('NewConfirmed', 'NewDeaths'):
        ax.set_yscale('log')

    # plt.xlim('2020-05-01', '2020-06-20')
    plt.savefig(gSaveBasePath + 'countries_' + column + '.png')
    # plt.show()


def plotCountryInfo2(all, column='Confirmed'):
    countriesNumbers = 8
    days = 30
    countries = all[-1][1]['Location'][1:countriesNumbers]
    # print('countries=', countries)
    # print('all=', all)

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for k, i in enumerate(countries):
        df = getCountryDayData(i, all)
        df = getCountryNewCasesAndDeathsDf(df)
        # print('df=', df)
        # df = binaryDf(df,labelAdd=False)
        df = df.iloc[-1 * days:, :]  # recent 30 days
        color = cm.jet(float(k) / countriesNumbers)
        # print('color=',color)
        title = column + ' Cases, Top ' + str(countriesNumbers) + ' countries, ' + 'Recent ' + str(days) + ' days'
        plotCountryAx(ax, df['Date'], df[column], label=i, title=title, color=color)

        # ax.text(df['Date'][-1], df[column][-1], i)
        # print(df.head(5))
        # print(df[column])
        ax.text(df['Date'].iloc[-1], df[column].iloc[-1], i, color=color)
        # break

    bottom, top = plt.ylim()
    bottom = max(bottom, 0)
    # print('bottom, top =',bottom, top)
    inter = (top - bottom) // 25
    # if column=='Confirmed':
    #     inter = 1000000
    for y in range(int(bottom), int(top), int(inter)):
        plt.plot(df['Date'], [y] * len(df['Date']), "--", lw=0.5, color="black", alpha=0.3)

    # plt.xlim('2020-05-01', '2020-06-20')
    # plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(gSaveBasePath + 'countries0_' + column + '.png')
    # plt.show()


def plotCountryInfo3(all, column='Confirmed'):
    countriesNumbers = 8
    countries = all[-1][1]['Location'][1:countriesNumbers]
    # print(countries)

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for k, i in enumerate(countries):
        df = getCountryDayData(i, all)
        df = getCountryNewCasesAndDeathsDf(df)
        # df = binaryDf(df,labelAdd=False)
        color = cm.jet(float(k) / countriesNumbers)

        title = column + ' Cases, Top ' + str(countriesNumbers) + ' countries'

        plotCountryAxBar(ax, df['Date'], df[column], label=i, title=title, color=color)
        # ax.text(df['Date'].iloc[-1], df[column].iloc[-1], i, color=color)

    bottom, top = plt.ylim()
    # print('bottom, top =',bottom, top)
    inter = 10000
    if column == 'Confirmed':
        inter = 1000000
    for y in range(int(bottom), int(top), inter):
        plt.plot(df['Date'], [y] * len(df['Date']), "--", lw=0.5, color="black", alpha=0.3)

    # plt.xlim('2020-05-01', '2020-06-20')
    # plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    # ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(gSaveBasePath + 'countries1_' + column + '.png')
    plt.show()


def plotCountryAx(ax, x, y, label, title=None, color=None):
    fontsize = 7
    ax.plot(x, y, label=label, c=color)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    # plt.show()


def plotCountryAxBar(ax, x, y, label, title, color=None):
    fontsize = 7
    ax.bar(x, y, label=label, color=color)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper left')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    # plt.show()


def getCountryDayData(country, allList):
    pdCountry = pd.DataFrame()
    for i in allList:
        date = i[0]
        df = i[1]

        countryLine = df[df['Location'] == country]
        # countryLine['Date'] = date
        try:
            countryLine.insert(1, "Date", [date], True)
            # print('countryLine=',countryLine)
            pdCountry = pdCountry.append(pd.DataFrame(data=countryLine))
        except BaseException:
            # print('date,country=',date,country)
            # print('countryLine=',countryLine)
            pass

    pdCountry.set_index(["Location"], inplace=True)
    # print(pdCountry)
    return pdCountry


if __name__ == '__main__':
    csvpath = r'./data/'
    plotCountriesFromOurWorld()
