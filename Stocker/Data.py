import pandas as pd
import yfinance as yf
import numpy as np
pd.options.mode.chained_assignment = None


class DataHandler:
    def __init__(self, symbol, period, interval):
        self.symbol = symbol.upper()
        self.period = period
        self.interval = interval

    def __fetchDefaultDataSet(self):
        self.parameters = ['Close', 'Open', 'Low', 'High']
        self.df = yf.Ticker(self.symbol).history(
            period=self.period, interval=self.interval)[
            map(str.title, self.parameters)]
        # print(self.df.head())

    def __calculateMovingAverage(self, period):
        self.movingAveragePeriod = period
        self.df[f'MA{period}'] = self.df['Close'].rolling(period).mean()

    def __calculateDefaultMACD(self):
        EMA12 = self.df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        EMA26 = self.df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        MACD = EMA12 - EMA26
        triggerLine = MACD.ewm(span=9, adjust=False, min_periods=9).mean()
        macdDelta = MACD - triggerLine

        self.df['MACD'] = self.df.index.map(MACD)
        self.df['Trigger Line'] = self.df.index.map(triggerLine)
        self.df['∆ MACD'] = self.df.index.map(macdDelta)

    def __calculateDefaultRSI(self):
        windowLength = 14
        self.df['Diff'] = self.df.index.map(self.df['Close'].diff(1))
        self.df['Gain'] = self.df['Diff'].clip(lower=0).round(2)
        self.df['Loss'] = self.df['Diff'].clip(upper=0).abs().round(2)
        self.df['Average Gain'] = self.df['Gain'].rolling(window=windowLength, min_periods=windowLength).mean()[
                                  :windowLength + 1]
        self.df['Average Loss'] = self.df['Loss'].rolling(window=windowLength, min_periods=windowLength).mean()[
                                  :windowLength + 1]

        # Average Gains
        for i, row in enumerate(self.df['Average Gain'].iloc[windowLength + 1:]):
            self.df['Average Gain'].iloc[i + windowLength + 1] = \
                (self.df['Average Gain'].iloc[i + windowLength] *
                 (windowLength - 1) +
                 self.df['Gain'].iloc[i + windowLength + 1]) \
                / windowLength
        # Average Losses
        for i, row in enumerate(self.df['Average Loss'].iloc[windowLength + 1:]):
            self.df['Average Loss'].iloc[i + windowLength + 1] = \
                (self.df['Average Loss'].iloc[i + windowLength] *
                 (windowLength - 1) +
                 self.df['Loss'].iloc[i + windowLength + 1]) \
                / windowLength

        self.df['RS'] = self.df['Average Gain'] / self.df['Average Loss']
        self.df['RSI'] = 100 - (100 / (1.0 + self.df['RS']))

    def __calculateIndicators(self):
        self.__calculateMovingAverage(30)
        self.__calculateDefaultMACD()
        self.__calculateDefaultRSI()

    def __formatDataset(self):
        self.df.dropna(inplace=True)
        self.df.drop(columns=['Trigger Line', 'Diff', 'Gain', 'Loss', 'Average Gain', 'Average Loss', 'RS'],
                     inplace=True)

    def __splitDataset(self):
        n = len(self.df)
        self.validate_df = self.df[int(n * 0.1):int(n * 0.3)]
        self.test_df = self.df[0:int(n * 0.1)]
        self.train_df = self.df[int(n * 0.3):]
        print(f"\nFull dataset shape:  {self.df.shape}")
        print(f"Train dataset shape: {self.train_df.shape}")
        print(f"Valid dataset shape: {self.validate_df.shape}")
        print(f"Test dataset shape:  {self.test_df.shape}\n")

    def __normalizeDatasets(self):
        self.train_mean = self.train_df.mean()
        self.train_std = self.train_df.std()
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.validate_df = (self.validate_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std
        self.df_std = (self.df - self.train_mean) / self.train_std

    def denormalizeDataset(self, dataSet):
        dataSet = dataSet * self.train_std + self.train_mean
        return dataSet

    def getDataSet(self):
        self.__fetchDefaultDataSet()
        self.__calculateIndicators()
        self.__formatDataset()
        return self.df

    def recalculateIndicators(self, dataSet):
        self.df = dataSet
        self.__calculateIndicators()
        return self.df

    def getNormalizedSplitDataSets(self):
        if self.df.empty:
            self.__fetchDefaultDataSet()
            self.__calculateIndicators()
            self.__formatDataset()
        self.__splitDataset()
        self.__normalizeDatasets()

        return self.df_std, self.train_df, self.validate_df, self.test_df

    def calculateActions_MA(self, df):
        df['Signal'] = 0
        df['Signal'] = np.where((df['MA30'] - df['Close']) >= 0, -1, 1)
        df['action'] = df['Signal'].diff()
        df['action'] = np.where(df['Signal'] != 0, df['action'], 0)

        actions = df.loc[(df['action'] != 0)]
        actions = actions.drop(['Open', 'Low', 'High', 'MACD', 'Trigger Line', 'MA30', 'Diff',
                                'Gain', 'Loss', 'Average Gain', 'Average Loss', 'RS'], axis=1)
        actions = actions.dropna()

        buy = actions.loc[(actions['action'] > 0)]
        sell = actions.loc[(actions['action'] < 0)]

        return df, buy, sell

    def calculateActions_MACDRSI(self, df):
        df['Signal'] = 0
        df['Signal'] = np.where(df['∆ MACD'] >= 0, 1, -1)
        df['Signal'] = np.where(abs(df['∆ MACD']) > 0.10, df['Signal'], 0)
        df['Signal'] = np.where((df['RSI'] > 45) & (df['Signal'] == 1), 2, df['Signal'])
        df['Signal'] = np.where((df['RSI'] < 55) & (df['Signal'] == -1), -2, df['Signal'])
        df['Signal'] = np.where((abs(df['Signal']) < 2), 0, df['Signal'])
        df['action'] = df['Signal'].diff()
        df['action'] = np.where(df['Signal'] != 0, df['action'], 0)
        actions = df.loc[(df['action'] != 0)]
        actions = actions.dropna()
        actions = actions.drop(['Open', 'Low', 'High', 'MACD', 'Trigger Line', 'MA30', 'Diff',
                                'Gain', 'Loss', 'Average Gain', 'Average Loss', 'RS'], axis=1)

        buy = actions.loc[(actions['action'] > 0)]
        sell = actions.loc[(actions['action'] < 0)]

        return df, buy, sell