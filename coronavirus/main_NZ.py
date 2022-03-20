"""
# python3 unicode
# author:Steven Huang 07/25/20
# function: Query NZ COVID-19 from https://www.health.govt.nz/
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# usgae:
# python ./mainNZ.py
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from lxml import etree

from plot_cases import gSaveBasePath, read_csv, init_folder
from common.get_html import openUrl, download_webfile
from common_path import create_path, get_file_name


MAIN_URL = 'https://www.health.govt.nz/'
URL = MAIN_URL + 'our-work/diseases-and-conditions/covid-19-novel-coronavirus/covid-19-data-and-statistics/covid-19-case-demographics'
CASE_CSV = 'https://raw.githubusercontent.com/minhealthnz/nz-covid-data/main/cases/covid-cases.csv'


def getDataFileFromWeb(url=URL):
    html = openUrl(url)
    # print(html)
    html = etree.HTML(html)
    # X = '//*[@id="node-10866"]/div/div/div/ul[2]/li[1]/a'
    # X = '//*[@id="node-10866"]/div[2]/div/div/p[13]/a'
    X = '//*[@id="case-details-csv-file"]'
    res = html.xpath(X)
    # print(len(res), res)
    if len(res) > 0:
        # print(res[0].get('href'))
        return MAIN_URL + res[0].get('href')[1:]  # remove the first '/'
    return None


def plotStatistcs(df, title, label):
    fontsize = 8
    kind = 'bar'
    # if df.shape[0] > 25:
    #     kind='barh'
    ax = df.plot(kind=kind, legend=False)  # color='gray'

    x_offset = -0.10
    y_offset = 3.8
    for p in ax.patches:
        b = p.get_bbox()
        val = "{}".format(int(b.y1 + b.y0))
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), fontsize=fontsize)

    ax.set_title(title, fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), rotation=30, fontsize=fontsize)
    plt.xlabel('')
    plt.ylabel('')
    plt.subplots_adjust(left=0.2, bottom=0.22, right=0.98, top=0.94, wspace=None, hspace=None)
    plt.savefig(gSaveBasePath + 'NZ_' + label + '.png')
    # plt.show()


def parseConfirmed(df):
    print('Confirmed dataset:\n', df.head())
    Sex = list(set(df['Sex']))
    AgeGroup = list(set(df['Age group'].astype(str)))
    print('AgeGroup:\n', AgeGroup)
    AgeGroup.sort()
    # AgeGroup = [ '<1', '1 to 4', '5 to 9', '10 to 14', '15 to 19', '20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70+']

    DHB = list(set(df['DHB']))
    bOverseas = list(set(df['Overseas travel']))

    if ' ' in bOverseas:
        bOverseas.remove(' ')

    print('Sex=', Sex)
    print('AgeGroup=', AgeGroup)
    print('DHB=', DHB)
    print('bOverseas=', bOverseas)
    # print('LastTravelCountry=', LastTravelCountry)

    columns = ['Gender', 'Number']
    dfSex = pd.DataFrame()
    for i in Sex:
        line = pd.DataFrame([[i, df[df['Sex'] == i].shape[0]]], columns=columns)
        dfSex = pd.concat([dfSex,line], ignore_index=True)
    dfSex.set_index(["Gender"], inplace=True)

    columns = ['Group', 'Number']
    dfAgeGroup = pd.DataFrame()
    for i in AgeGroup:
        line = pd.DataFrame([[i, df[df['Age group'] == i].shape[0]]], columns=columns)
        dfAgeGroup = pd.concat([dfAgeGroup,line], ignore_index=True)
    dfAgeGroup.set_index(["Group"], inplace=True)

    columns = ['DHB', 'Number']
    dfDHB = pd.DataFrame()
    for i in DHB:
        line = pd.DataFrame([[i, df[df['DHB'] == i].shape[0]]], columns=columns)
        dfDHB = pd.concat([dfDHB,line], ignore_index=True)
    dfDHB.set_index(["DHB"], inplace=True)

    columns = ['Overseas', 'Number']
    dfbOverseas = pd.DataFrame()
    for i in bOverseas:
        line = pd.DataFrame([[i, df[df['Overseas travel'] == i].shape[0]]], columns=columns)
        dfbOverseas = pd.concat([dfbOverseas,line], ignore_index=True)
    dfbOverseas.set_index(["Overseas"], inplace=True)

    # dfSex = dfSex.sort_values(by=0, axis=1) #dfSex.sort_values(by=['Female'], ascending=False)
    # dfAgeGroup = dfAgeGroup.sort_values(by=['Case_Per_1M_people'], ascending=False)
    dfDHB = dfDHB.sort_values(by=['Number'], ascending=False)
    # dfbOverseas = dfbOverseas.sort_values(by=['Case_Per_1M_people'], ascending=False)
    # dfLastTravelCountry = dfLastTravelCountry.sort_values(by=['Number'], ascending=False)

    # print(dfSex)
    # print(dfAgeGroup)
    # print(dfDHB)
    # print(dfbOverseas)
    # print(dfLastTravelCountry)

    now = datetime.datetime.now()
    today = str(' Date:') + str(now.strftime("%Y-%m-%d %H:%M:%S"))
    label = 'Gender'
    plotStatistcs(dfSex, label=label, title=label + ' ' + today)
    label = 'AgeGroup'
    plotStatistcs(dfAgeGroup, label=label, title=label + ' ' + today)
    label = 'DHB'
    plotStatistcs(dfDHB, label=label, title=label + ' ' + today)
    label = 'Overseas'
    plotStatistcs(dfbOverseas, label=label, title=label + ' ' + today)
    # label='LastTravelCountry'
    # plotStatistcs(dfLastTravelCountry, label=label, title=label + ' ' + today)
    plt.show()


def plotTotal(df, title, label, showNumberOnBar=False):
    fontsize = 8
    plt.figure()
    ax = df.plot(kind='bar', legend=False)

    if showNumberOnBar:
        x_offset = -0.2
        y_offset = 0.1
        for p in ax.patches:
            b = p.get_bbox()
            val = "{}".format(int(b.y1 + b.y0))
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), fontsize=fontsize)

    ax.set_title(title, fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), rotation=30, fontsize=fontsize)
    plt.xlabel('')
    plt.ylabel('')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(gSaveBasePath + label + '.png')
    # plt.show()


def plotNZDataChange(df):
    def getDataRecordNum(df, date):
        return df[df['Report Date'] == date].shape[0]

    def totalDays(start, stop):
        delta = stop - start
        # print(delta.days)  # type(days)
        return delta.days

    DATEFORMAT = r'%Y-%m-%d'  # r'%d/%m/%Y'
    # print(df.head())
    # pd.to_datetime(df['Report Date'])

    dfDate = df['Report Date']
    dfDate = pd.to_datetime(dfDate)  # format=DATEFORMAT
    # print('dtypes=', dfDate.dtypes)
    # print(dfDate.shape)

    dfDate = sorted(set(dfDate))
    # print('dfDate=', len(dfDate),dfDate)

    startDate = dfDate[0]
    stopDate = dfDate[-1]
    days = totalDays(startDate, stopDate)
    # print('startDate,stopDate=',startDate,stopDate,days)

    columns = ['Date', 'Number', 'Cumlative']
    dfStat = pd.DataFrame()
    s = 0
    for i in range(days + 1):
        d = startDate + datetime.timedelta(days=i)
        d = datetime.datetime.strftime(d, DATEFORMAT)
        number = getDataRecordNum(df, d)
        # print(d, number)
        s += number
        line = pd.DataFrame([[d, number, s]], columns=columns)
        dfStat = pd.concat([dfStat,line], ignore_index=True)

    now = datetime.datetime.now()
    today = str(' Date:') + str(now.strftime("%Y-%m-%d %H:%M:%S"))

    dfStat.set_index(["Date"], inplace=True)
    # print('dfStat=', dfStat)

    recentDays = 40
    dfStatRecent = dfStat[-1 * recentDays:]

    dfStat = dfStat.iloc[::6]  # drop some lines
    # dfWorld = binaryDf(dfWorld,False) #drop half
    print('head after drop:\n', dfStat.head())

    label = 'NZ_COVID-19_EveryDayCases'
    plotTotal(dfStat['Number'], label=label, title=label + ' ' + today)

    label = 'NZ_COVID-19_CumlativeCases'
    plotTotal(dfStat['Cumlative'], label=label, title=label + ' ' + today)
    # print(dfStat['Number'][-30:])

    label = 'NZ_COVID-19_RecentCases'
    title = label + ' ' + str(recentDays) + ' days, ' + today
    plotTotal(dfStatRecent['Number'], label=label, title=title, showNumberOnBar=True)
    plt.show()


def download_file(url):
    name = get_file_name(url)
    print('url=', url)
    print('name=', name)

    dst_path = r'./NZ/'
    create_path(dst_path)
    file = os.path.join(dst_path, name)
    res = download_webfile(url, file)
    if not res:
        print("Download file failed, please check the url!")
        return None
    return file


def getNZCovid19():
    # file_url = getDataFileFromWeb()
    # if file_url is None:
    #     print("Can't find the file url form website, something wrong!")
    #     return None

    file_url = CASE_CSV
    file = download_file(file_url)
    return read_csv(file)


def plotStatistic(df):
    parseConfirmed(df)
    plotNZDataChange(df)


def main():
    init_folder()
    df = getNZCovid19()
    # df = read_csv(r'.\NZ\covid_cases_2022-01-26.csv')

    if df is not None:
        plotStatistic(df)


if __name__ == '__main__':
    main()
