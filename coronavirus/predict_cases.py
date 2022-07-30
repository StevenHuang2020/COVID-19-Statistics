"""
# python3 steven
# LSTM regression, solve data set with time change
"""
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.utils import shuffle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score

# plt.rcParams['savefig.dpi'] = 300 #matplot figure quality when save
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # not print tf debug info

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, initializers

from json_update import get_datetime
from common_path import traverse_files, create_path

gDatasetPath = r'.\data\OurWorld'
gSaveBasePath = r'..\images'
gSaveChangeData = r'.\data\dataChange'
gSaveCountryData = r'.\data\dataCountry'
gSavePredict = r'.\data\dataPredict'

gScaler = MinMaxScaler()  # StandardScaler()
gPredictDays = 15


def preprocessDb(dataset):
    dataset = gScaler.fit_transform(dataset)
    # print('dataset=', dataset[:5])
    return dataset


def create_dataset(dataset, look_back=1):
    """convert an array of values into a dataset matrix """
    dataset = dataset.flatten()
    # print(dataset.shape)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def get_dataset(file):
    if not os.path.exists(file):
        print('warning, data file is not exist.')
        return None

    dataset = pd.read_csv(file)
    # dataset = pd.read_csv(file,sep=',', encoding='utf-8')
    dataset = dataset[dataset['location'] == 'World']
    # dataset = dataset.rename(columns={"Total confirmed cases of COVID-19 (cases)": "Cases"})
    print(dataset.head())
    dataset = dataset.loc[:, ['date', 'total_cases']]
    dataset = dataset.rename(columns={'date': 'Date', "total_cases": "Cases"})
    # dataset = dataset['Date', 'Cases']

    dataset = dataset.dropna()

    # print(dataset.describe().T)
    print(dataset.head())
    print(dataset.tail())
    print(dataset.shape)
    print(dataset.dtypes)
    return dataset


def plotDataAx(ax, x, y, label='', fontsize=5, color=None):
    ax.plot(x, y, label=label, color=color)
    # ax.set_aspect(1)
    ax.legend(fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize, fontweight=10)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.subplots_adjust(left=0.02, bottom=0.09, right=0.99, top=0.92, wspace=None, hspace=None)


def predictFuture(model, start, Number=5):
    # print('---------------future---------')
    # print('start=', start)
    # print(start.shape,'startval:', gScaler.inverse_transform(start.reshape(1, -1)))
    # start = np.array([start]).reshape(1, 1, 1) #lookback  tf 2.8
    start = np.array([start]).reshape(1, 1) #lookback  tf 2.9
    result = []
    result.append(start.flatten()[0])
    for _ in range(Number):
        nextPred = model.predict(start)
        # print(nextPred)
        result.append(nextPred.flatten()[0])
        start = nextPred
    # print('predict value=', result)
    result = gScaler.inverse_transform(np.array(result).reshape(1, -1)).flatten()
    result = list(map(int, result))
    # print('after inverse redict value=', result)
    return result


def plotPredictCompare(model, trainX, index, data):
    trainPredict = model.predict(trainX).flatten()
    trainPredict = gScaler.inverse_transform(trainPredict.reshape((trainPredict.shape[0], 1))).flatten()

    # print('trainPredict.shape=', trainPredict.shape, index.shape, data.shape)
    data = data.flatten()
    # print(index.shape)
    # print(trainPredict.shape)
    # print(data.shape)
    # print('raw=',data)
    # print('pred=',trainPredict)

    offset = 70  # 120
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1)
    plt.title('PredictTime: ' + get_datetime())
    plotDataAx(ax, index[offset + 2:-1], data[offset + 2:-1], 'True Data')

    pre_y = trainPredict[offset + 1:-1]
    index_x = index[offset + 2: offset + 2 + len(pre_y)]
    plotDataAx(ax, index_x, pre_y, 'Prediction')
    plt.savefig(os.path.join(gSaveBasePath, 'WorldPredictCompare.png'))
    plt.show()


def changeNewIndexFmt(newIndex, src_fmt='%d/%m/%Y', dst_fmt='%Y-%m-%d'):
    new = []
    for i in newIndex:
        i = datetime.datetime.strptime(i, src_fmt)
        i = datetime.datetime.strftime(i, dst_fmt)  # '%b %d, %Y'
        new.append(i)
    return new


def plotPredictFuture(model, trainY, index, data, days=gPredictDays):
    pred = predictFuture(model, trainY[-1], days)
    print('predict start date:', index[-1])

    startIndex = index[-1]
    fmt = '%d/%m/%y'
    if '-' in startIndex:
        fmt = '%Y-%m-%d'
    sD = datetime.datetime.strptime(startIndex, fmt)  # '%Y-%m-%d'

    newIndex = []
    # dst_fmt = '%d/%m/%Y'
    startIndex = datetime.datetime.strftime(sD, fmt)
    newIndex.append(startIndex)
    for i in range(days):
        d = sD + datetime.timedelta(days=i + 1)
        d = datetime.datetime.strftime(d, fmt)
        # print(d)
        newIndex.append(d)
    print('predict period:', newIndex)

    df = pd.DataFrame({'Date': newIndex, 'Predicted cases': pred})

    # add predict day newCases
    df['Predicted daily newCases'] = 0
    for i in range(1, df.shape[0]):
        df.iloc[i, 2] = df.iloc[i, 1] - df.iloc[i - 1, 1]

    print('table:\n', df)

    startIndex = datetime.datetime.strptime(startIndex, fmt)
    predictTime = datetime.datetime.strftime(startIndex, '%Y-%m-%d')
    file = os.path.join(gSavePredict, predictTime + '_predict.csv')
    df.to_csv(file, index=True)

    offset = 150  # 70 120
    # plt.figure(figsize=(8, 6))
    plt.title('Future ' + str(days) + ' days Covid-19,' + ' Predicted time: ' + get_datetime())

    ax = plt.gca()
    # ax = plt.subplot(1, 1, 1)
    plotDataAx(ax, index[offset:], data[offset:], 'Now cases')

    newIndex = changeNewIndexFmt(newIndex, fmt)
    plotDataAx(ax, newIndex, pred, 'Predicted cases')
    # print('oldIndex=', index[offset:])
    # print('newIndex=', newIndex)

    tb = plt.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    # colList = list(range(len(df.columns)))
    colList = [2]
    tb.auto_set_column_width(col=colList)

    # plt.axis('off')
    plt.savefig(os.path.join(gSaveBasePath, 'WorldFuturePredict.png'))
    plt.show()


def create_model(look_back=1):
    model = Sequential()
    model.add(tf.keras.Input(shape=(1, look_back)))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # fixed learning rate
    # lr = 1e-3

    # learning rate schedule
    lr = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                               decay_steps=10000,
                                               decay_rate=0.99)

    # opt = optimizers.SGD(learning_rate=lr)
    # opt = optimizers.SGD(learning_rate=lr, momentum=0.8, nesterov=False)
    # opt = optimizers.RMSprop(learning_rate=lr, rho=0.9, epsilon=1e-08)
    opt = optimizers.Adam(learning_rate=lr)
    # opt = optimizers.Adadelta(learning_rate=lr)
    # opt = optimizers.Adagrad(learning_rate=lr)
    # opt = optimizers.Adamax(learning_rate=lr)
    # opt = optimizers.Nadam(learning_rate=lr)
    # opt = optimizers.Ftrl(learning_rate=lr)

    model.compile(optimizer=opt, loss='mse')
    # model.summary()
    return model


def prepareDataset(dataset, look_back):
    index = dataset.iloc[:, 0].values
    rawdata = dataset.iloc[:, 1].values
    data = rawdata.reshape((rawdata.shape[0], 1))
    # print('raw=',rawdata[-5:])
    data = preprocessDb(data)  # scaler features
    # print('raw=',rawdata[-5:])
    # print('index=',index[-5:])
    X, Y = create_dataset(data, look_back)

    # X, Y = shuffle(X, Y, random_state=0)
    # print('X =', X[:5])
    # print('Y =', Y[:5])

    # X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    # Y = np.reshape(X, (Y.shape[0], 1, 1))

    print('X.shape =', X.shape)
    print('Y.shape =', Y.shape)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
    return X, Y, index, rawdata


def train(dataset, first=False, look_back=1):
    x_train, y_train, index, rawdata = prepareDataset(dataset, look_back)
    print('x_train.shape =', x_train.shape)
    # print('y_train.shape =', y_train.shape)

    pre_train_path = r'pre_trained'
    if first or not os.path.exists(pre_train_path):
        print('\nTraining model first time...\n')
        model = create_model(look_back)
    else:
        print('\nTraining model from pre-trained weights...\n')
        model = tf.keras.models.load_model(pre_train_path)

        lr = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-6,
            decay_steps=100,
            decay_rate=0.9)

        model.optimizer.learning_rate = lr  # 1e-5

    model.summary()
    model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=1)  # verbose=2

    model.save(pre_train_path)

    # a = np.array([trainY[-1]]).reshape(-1,1,1)
    # a = np.array([[0.88964097]]).reshape(-1,1,1)
    # a = np.array([0.6]).reshape(1,1,1)
    # print('predict=', model.predict(a))

    # -----------------start plot--------------- #
    plotPredictCompare(model, x_train, index, rawdata)
    plotPredictFuture(model, y_train, index, rawdata)


def getPredictDf(file):
    df = pd.read_csv(file)
    # df['Date'] = pd.to_datetime(df.Date, format='%m %d, %Y')
    # df.set_index(["Date"], inplace=True)
    return df


def evaulate_predition(df, file):
    def getTrueCases(day, df):
        for i in range(df.shape[0]):
            d = df.iloc[i, df.columns.get_loc('Date')]
            cases = df.iloc[i, df.columns.get_loc('Cases')]
            # d = datetime.datetime.strptime(d,'%b %d, %Y')
            d = datetime.datetime.strptime(d, '%Y-%m-%d')
            # print('day=', day)
            if '-' not in day:
                date = datetime.datetime.strptime(day, '%m/%d/%Y')
            else:
                date = datetime.datetime.strptime(day, '%Y-%m-%d')
            # print(d, date, cases)
            if date == d:
                return cases
        return 0

    dfPredict = getPredictDf(file)
    dfPredict.drop(['Predicted daily newCases'], axis=1, inplace=True)

    predictTime = file[file.rfind('\\') + 1: file.rfind('_')]

    print('dfPredict=\n', dfPredict)
    print('predictTime=', predictTime)
    allCases = np.zeros((dfPredict.shape[0],))
    accs = []
    for i in range(dfPredict.shape[0]):
        # date = dfPredict.iloc[i,0]
        # predictCase = dfPredict.iloc[i,1]
        date = dfPredict.loc[i]['Date']
        predictCase = dfPredict.loc[i]['Predicted cases']

        acc = 0
        cases = getTrueCases(date, df)
        if cases != 0:
            acc = round((1 - (np.abs(cases - predictCase) / cases)), 5)
            acc = format(acc, '.5f')

        # print(date,predictCase)
        # print(date,predictCase,cases)

        accs.append(acc)
        allCases[i] = cases
        # break

    dfPredict['Cases'] = allCases
    dfPredict['Precision'] = accs
    dfPredict = dfPredict.iloc[:, 1:]  # remove index number column

    dfPredict.Cases = dfPredict.Cases.astype('int64')

    if '-' not in dfPredict['Date'].values[0]:  # 05/09/2022 ==> 2022-04-29
        newDates = changeNewIndexFmt(dfPredict['Date'].values, '%m/%d/%Y')
        dfPredict.loc[:, 'Date'] = newDates
        # print('dfPredict=\n', dfPredict)

    # plt.figure(figsize=(8, 6))
    title = 'Prediction Precision\n' + 'PredictedTime: ' + predictTime + ' CheckTime: ' + get_datetime()
    plt.title(title, fontsize=9)
    tb = plt.table(cellText=dfPredict.values, colLabels=dfPredict.columns, loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    # colList = list(range(len(dfPredict.columns)))
    colList = [1, 2]
    tb.auto_set_column_width(col=colList)

    # set precision colomn font color
    # print('dfPredict.values.shape=', dfPredict.values.shape)
    for i in range(1, dfPredict.values.shape[0] + 1):
        tb[(i, 3)].get_text().set_color('red')

    plt.axis('off')
    plt.savefig(os.path.join(gSaveBasePath, 'WorldFuturePredictPrecise.png'))
    plt.show()


def getNewestFile(path, fmt='csv', index=-1):
    file_dict = {}
    for i in traverse_files(path, fmt):
        ctime = os.stat(os.path.join(i)).st_mtime  # st_ctime
        ctime = datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d-%H:%M')
        file_dict[ctime] = i

    sort = sorted(file_dict.keys())
    # print('file_dict=', file_dict)
    # print('sort=', sort)
    # get a previrous predicted file
    if abs(index) >= len(sort):
        index = 0

    return file_dict[sort[index]]


def predict(data):
    if data is None:
        return

    train(data)

    file = getNewestFile(gSavePredict, index=-1 * gPredictDays + 5)
    print('Last predicted file:', file)
    evaulate_predition(data, file)


def main():
    print('\nTensorflow version: ', tf.version.VERSION)
    create_path(gSavePredict)

    file = os.path.join(gDatasetPath, 'owid-covid-data.csv')
    data = get_dataset(file)
    predict(data)


if __name__ == '__main__':
    main()
