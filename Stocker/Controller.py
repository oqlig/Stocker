import pandas
import yfinance as yf
import tensorflow as tf
import numpy as np
from time import sleep
from Stocker.Data import DataHandler
from Stocker.Window import DataWindow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


MAX_EPOCHS = 50
CONV_WIDTH = 3
LABEL_WIDTH = 30
OUTPUT_STEPS = 5
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


def compileAndFit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
    return history


class AppController:
    def __init__(self):
        self.__stock = ''
        self.__interval = ''
        self.__timeFrame = ''
        self.__algorithm = ''
        self.__dataHandler: DataHandler
        self.__dataWindow: DataWindow
        self.__modelCNN: tf.keras.model
        self.__df_final: pandas.DataFrame
        self.__df_buy: pandas.DataFrame
        self.__df_sell: pandas.DataFrame

    def __showStock(self):
        if self.__stock != '':
            print(f'''    Stock: {self.__stock}''')
        else:
            print(f'''    Stock: ---''')

    def __showAlgorithm(self):
        if self.__algorithm != '':
            if self.__algorithm == 'alpha':
                if self.__interval == '1d':
                    print(f'''    Algorithm: MA30 - daily''')
                elif self.__interval == '1h':
                    print(f'''    Algorithm: MA30 - hourly''')
            else:
                print(f'''    Algorithm: MACD + RSI - hourly''')
        else:
            print(f'''    Algorithm: ---''')

    def __showStrategy(self):
        self.__showStock()
        self.__showAlgorithm()

    def __showWelcomingMessage(self):
        print('''\n    Welcome to Stocker!''')

    def __showAlgorithmMenu(self):
        print('''   
    Please choose a algorithm for trading:   
        1 - MA30 ........ ( daily ticks )
        2 - MA30 ........ ( hourly ticks )
        3 - MACD + RSI .. ( hourly ticks )''')

    def __checkStockAvailability(self, stockSymbol):
        stock = yf.Ticker(stockSymbol)
        if stock.info['regularMarketPrice'] is None:
            return False
        else:
            return True

    def __assignStock(self):
        print('''    Please enter a 4-digit stock symbol.''')
        stockSymbol = str(input('\n')).upper()
        if self.__checkStockAvailability(stockSymbol=stockSymbol):
            self.__stock = stockSymbol
            print(f'''    Stock symbol changed to {stockSymbol}''')
            sleep(3)
        else:
            print(f'''    Stock symbol {stockSymbol} unknown''')
            sleep(4)

    def __assignAlgorithm(self):
        self.__showAlgorithmMenu()
        choice = int(input('\n'))
        if choice == 1:
            self.__algorithm = 'alpha'
            self.__timeFrame = '10y'
            self.__interval = '1d'
            print(f'''    Algorithm changed to: 1 - MA30 ........ ( daily ticks )''')
            sleep(3)
        elif choice == 2:
            self.__algorithm = 'alpha'
            self.__timeFrame = '2y'
            self.__interval = '1h'
            print(f'''    Algorithm changed to: 2 - MA30 ........ ( hourly ticks )''')
            sleep(3)
        elif choice == 3:
            self.__algorithm = 'beta'
            self.__timeFrame = '2y'
            self.__interval = '1h'
            print(f'''    Algorithm changed to: 3 - MACD + RSI .. ( hourly ticks )''')
            sleep(3)
        else:
            print('''    Please enter a valid choice.''')
            sleep(4)

    def __predictAndPlot(self):
        print('''    Processing...''')
        self.__dataHandler = DataHandler(symbol=self.__stock, period=self.__timeFrame, interval=self.__interval)
        dataSet = self.__dataHandler.getDataSet()
        self.__columnIndices = {name: i for i, name in enumerate(dataSet.columns)}
        self.__numberOfFeatures = dataSet.shape[1]

        df_std, df_train_std, df_val_std, df_test_std = self.__dataHandler.getNormalizedSplitDataSets()
        self.__dataWindow = DataWindow(30, OUTPUT_STEPS, OUTPUT_STEPS, df_train_std, df_val_std, df_test_std, ['Close'])

        clearConsole()

        self.__modelCNN = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                                               tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
                                               tf.keras.layers.Dense(OUTPUT_STEPS * self.__numberOfFeatures,
                                                                     kernel_initializer=tf.initializers.zeros()),
                                               tf.keras.layers.Reshape([OUTPUT_STEPS, self.__numberOfFeatures])])
        modelConv_MultiHistory = compileAndFit(self.__modelCNN, self.__dataWindow)

        clearConsole()
        print('''   Running Algorithm''')

        set_test = tf.convert_to_tensor(df_std.tail(30))
        set_test = tf.reshape(set_test, [1, 30, 8])
        future = self.__modelCNN.predict(set_test).flatten()

        df_std = df_std.append({'Close': future[0]}, ignore_index=True)
        df_std = df_std.append({'Close': future[8]}, ignore_index=True)
        df_std = df_std.append({'Close': future[16]}, ignore_index=True)
        df_std = df_std.append({'Close': future[24]}, ignore_index=True)
        df_std = df_std.append({'Close': future[32]}, ignore_index=True)

        output = self.__dataHandler.denormalizeDataset(df_std.tail(200))
        output = self.__dataHandler.recalculateIndicators(output)
        self.__df_final = output.tail(100)

        if self.__algorithm == 'alpha':
            self.__df_final, self.__df_buy, self.__df_sell = self.__dataHandler.calculateActions_MA(self.__df_final)
        else:
            self.__df_final, self.__df_buy, self.__df_sell = self.__dataHandler.calculateActions_MACDRSI(self.__df_final)

        future = self.__df_final.tail(6)
        past = self.__df_final.head(95)
        symbol = self.__stock
        if self.__interval == '1d':
            myPeriod = '100 days'
            myInterval = '1d'
        else:
            myPeriod = '100 hours'
            myInterval = '1h'

        clearConsole()
        print('''   Plotting...''')

        figure = make_subplots(rows=3, cols=1,
                               row_heights=[0.5, 0.25, 0.25],
                               subplot_titles=(f"{symbol} - Stock Price",
                                               f"{symbol} - MACD",
                                               f"{symbol} - RSI"),
                               vertical_spacing=0.1)

        # Future Price
        figure.append_trace(
            go.Scatter(
                x=future.index,
                y=future['Close'],
                line=dict(color='orange', width=5),
                name=f'{symbol} - Predicted Price',
                mode='lines',
                legendgroup='1',
                legendgrouptitle_text=f"{symbol} - Stock Price",
            ), row=1, col=1
        )
        #
        # buy
        figure.append_trace(
            go.Scatter(
                x=self.__df_buy.index,
                y=self.__df_buy['Close'],
                mode='markers',
                marker=dict(color='#07fc03', size=10, line=dict(width=1.5, color='DarkSlateGrey')),
                name=f'{symbol} - Buy',
                # showlegend=False,
                legendgroup='1',
                legendgrouptitle_text=f"{symbol} - Stock Price",
            ), row=1, col=1
        )

        # sell
        figure.append_trace(
            go.Scatter(
                x=self.__df_sell.index,
                y=self.__df_sell['Close'],
                mode='markers',
                marker=dict(color='#fc0303', size=10, line=dict(width=1.5, color='DarkSlateGrey')),
                name=f'{symbol} - Sell',
                # showlegend=False,
                legendgroup='1',
            ), row=1, col=1
        )

        # MA__
        figure.append_trace(
            go.Scatter(
                x=self.__df_final.index,
                y=self.__df_final[f'MA30'],
                line=dict(color='#b836bf', width=2),
                name=f'MA30',
                # showlegend=False,
                legendgroup='1',
            ), row=1, col=1
        )

        # price Candlestick
        figure.append_trace(
            go.Candlestick(
                x=past.index,
                open=past['Open'],
                high=past['High'],
                low=past['Low'],
                close=past['Close'],
                name=f'{symbol} - candle',
                # showlegend=False
                legendgroup='1',
            ), row=1, col=1
        )

        # MACD
        figure.append_trace(
            go.Scatter(
                x=self.__df_final.index,
                y=self.__df_final['MACD'],
                line=dict(color='#ffa352', width=2),
                name='MACD',
                # showlegend=False,
                legendgroup='3',
                legendgrouptitle_text=f"{symbol} - MACD",
            ), row=2, col=1
        )

        # Trigger Line
        figure.append_trace(
            go.Scatter(
                x=self.__df_final.index,
                y=self.__df_final['Trigger Line'],
                line=dict(color='#5274ff', width=2),
                # showlegend=False,
                legendgroup='3',
                name='Trigger Line'
            ), row=2, col=1
        )

        # Colorize the histogram values
        colors = np.where(self.__df_final['∆ MACD'] < 0, '#f54242', '#55ad2f')
        # Plot the histogram
        figure.append_trace(
            go.Bar(
                x=self.__df_final.index,
                y=self.__df_final['∆ MACD'],
                name='∆ MACD',
                marker_color=colors,
                legendgroup='3',
            ), row=2, col=1
        )

        # Make RSI Plot
        figure.add_trace(go.Scatter(
            x=self.__df_final.index,
            y=self.__df_final['RSI'],
            line=dict(color='#ffa352', width=2),
            name='RSI',
            legendgroup='4',
            legendgrouptitle_text=f"{symbol} - RSI",
        ), row=3, col=1
        )

        figure.update_yaxes(range=[-10, 110], row=3, col=1)
        figure.add_hline(y=0, col=1, row=3, line_color="#666", line_width=2)
        figure.add_hline(y=100, col=1, row=3, line_color="#666", line_width=2)
        figure.add_hline(y=55, col=1, row=3, line_color='#336699', line_width=1, line_dash='dash')
        figure.add_hline(y=45, col=1, row=3, line_color='#336699', line_width=1, line_dash='dash')

        layout = go.Layout(
            title=f'{symbol} - Stock Price',
            plot_bgcolor='#efefef',
            font_family='Monospace',
            font_color='#000000',
            font_size=15,
            legend_tracegroupgap=40,
            yaxis=dict(
                title='Price [$]'
            ),
            xaxis=dict(
                title=f'Time ({myPeriod}) [{myInterval}]',
                showticklabels=False,
                rangeslider=dict(
                    visible=False
                )
            ),
            yaxis2=dict(
                title='∆ [$]'
            ),
            xaxis2=dict(
                title=f'Time ({myPeriod}) [{myInterval}]',
                showticklabels=False,
                rangeslider=dict(
                    visible=False
                )
            ),
            yaxis3=dict(
                title='RSI [%]'
            ),
            xaxis3=dict(
                title=f'Time ({myPeriod}) [{myInterval}]',
                showticklabels=False,
                rangeslider=dict(
                    visible=False
                )
            )
        )

        figure.update_layout(layout)
        figure.show()

    def startup(self):
        clearConsole()
        self.__showWelcomingMessage()
        while self.__stock == '':
            self.__showStrategy()
            self.__assignStock()
            clearConsole()
        while self.__algorithm == '':
            self.__showStrategy()
            self.__assignAlgorithm()
            clearConsole()

        self.__showStrategy()
        input('''\n    Press Enter/return to continue with prediction...''')
        clearConsole()

        self.__predictAndPlot()

        clearConsole()
        print('''    Complete! Thank You!''')

