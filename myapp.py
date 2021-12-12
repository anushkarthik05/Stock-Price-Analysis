import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.stats import norm
import warnings
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore") 
st.write("""
# Stock Price Prediction
Shown are the stock **closing price** and ***volume*** of Apple!
""")
key='6322cd72a9c28927a423b2e5e4b3775bc431eb73'
df=pdr.get_data_tiingo('AAPL',api_key=key)
df.to_csv('APPL.csv')
df=pd.read_csv('APPL.csv')
df2=df
st.write("""
### Head
""")
with st.echo():
    st.dataframe(df.head())
st.write("""
### Tail
""")
with st.echo():
    st.dataframe(df.tail())

import plotly.graph_objects as go
from datetime import datetime
import pandas_datareader.data as web
import plotly.express as px

with st.echo():
    st.write('OHLC Plot')
    fig1 = go.Figure(data=go.Ohlc(x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']))

    fig1.update_layout(xaxis_rangeslider_visible=False)
    #fig1.show()
    st.plotly_chart(fig1,use_container_width=True)



# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
#tickerSymbol = 'AAPL'
#get data on this ticker
#tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
#tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits
#st.table(tickerDf.head())
#st.write("""
## Closing Price
#""")
#st.line_chart(tickerDf.Close)
#st.write("""
## Volume Price
#""")
#st.line_chart(tickerDf.Volume)

with st.echo():
    st.write('Candlestick Plot')
    fig2 = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
    fig2.update_layout(xaxis_rangeslider_visible=False)
    #fig2.show()
    st.plotly_chart(fig2,use_container_width=True)

with st.echo():
    st.write('Area Plot')
    fig3 = px.area(df, x="date", y="close",)
    #fig3.show()
    st.plotly_chart(fig3,use_container_width=True)

with st.echo():
    st.write('Time Series Plot')
    fig4 = go.Figure([go.Scatter(x=df['date'], y=df['high'])])
    #fig4.show()
    st.plotly_chart(fig4,use_container_width=True)

st.write("""
## Adj Close values with respect to a set of days
""")

with st.echo():
    ma_day = [10, 20, 50]
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        df[column_name] = df['adjClose'].rolling(ma).mean()
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    axes.plot(df[['adjClose', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']])
    st.pyplot(fig)

st.write("""
## Daily Returns
""")

with st.echo():
    # We'll use pct_change to find the percent change for each day

    df['Daily Return'] = df['adjClose'].pct_change()

    # Then we'll plot the daily return percentage
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    axes.plot(df['Daily Return'])
    st.pyplot(fig)

st.write("""
### Shape of the dataset
""")

with st.echo():
    st.write(df.shape)


st.write("""
### Info of the dataset
""")
import io 
with st.echo():
    buffer = io.StringIO()
    df.info(buf=buffer)
    s=buffer.getvalue()
    st.text(s)

st.write("""
### Dropping Unnecessary columns 
""")

with st.echo():
    df.drop(['MA for 10 days','MA for 20 days','MA for 50 days','Daily Return'],inplace=True,axis=1)


st.write("""
### Describe the updated dataset 
""")

with st.echo():
    st.write(df.describe())

st.write("""
### Check for Null Values 
""")

with st.echo():
    st.write((df.isna().sum()/len(df))*100)

st.write("""
# Feature Engineering

### Handling Categorical values
""")

with st.echo():
    st.write(df['symbol'].unique())

st.write("""
### We observe that the column 'symbol' contains only one value. i,e. AAPL so label encoding will not benefit. 
Hence we can remove the entire column.
""")

with st.echo():
    df.drop(['symbol'],inplace=True,axis=1)

with st.echo():
    st.write(df.shape)

st.write("""
### Transformation of Date Format
""")

with st.echo():
    df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.date())
    st.dataframe(df)

st.write(f"Dataframe contains stock prices between {df.date.min()} {df.date.max()}")
st.write(f"Total days={(df.date.max()-df.date.min()).days} days")

st.write("""
### Drop Duplicates if present
""")

with st.echo():
    df.drop_duplicates()
    st.dataframe(df)
    st.write(df.shape)

st.write("""
### Correlation
""")

with st.echo():
    c_df = df
    c_df = c_df.drop(['adjClose'], axis = 1)
    st.write(c_df.shape)

with st.echo():
    corr=c_df.corr()
    fig = plt.figure(figsize=(15,15))
    mask = np.zeros_like(corr,dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr,mask=mask,annot=True)
    st.pyplot(fig)

with st.echo():
    temp = c_df.corr().style.background_gradient(cmap="coolwarm")
    st.dataframe(temp)

st.write("""
**The following are the column pairs with higher correlation than the threshold: ( Threshold : 0.9)**


*   close -> open
*   close -> low
*   close -> high
*   low -> high
*   open -> high
*   open -> low
*   adjhigh -> adjopen
*   adjhigh -> adjlow
*   adjopen -> adjlow
*   close -> volume
*   high -> volume
*   low -> volume
*   open -> volume

**We remove the following columns**

* open
* low
* volume
* adjlow
* adjopen

### Remove Unnecessary Columns
""")

with st.echo():
    df.drop(['adjLow','adjOpen','low','open','volume'],inplace=True,axis=1)
    st.write(df.shape)

st.dataframe(df)    

st.write("""
### Converting Datatypes
""")
with st.echo():
    st.text(df.dtypes)
with st.echo():
    df.drop(['date'],inplace = True,axis=1)

st.write("""
## Feature Scaling
### Normalization
""")

with st.echo():
    temp = df
    names = temp.columns

    scaler = preprocessing.MinMaxScaler()
    nn_ds = scaler.fit_transform(temp)
    nn_ds = pd.DataFrame(nn_ds, columns=names)

    st.dataframe(nn_ds)

st.write("""
### Standardization
""")

with st.echo():
    from sklearn.preprocessing import StandardScaler
    names = nn_ds.columns
    ns_ds = StandardScaler().fit_transform(nn_ds) 
    ns_ds = pd.DataFrame(data=ns_ds, columns=names)
    st.dataframe(nn_ds)

st.write("""
### Skewness - Outliers - Kurtosis
""")

with st.echo():
    skew_features = ns_ds.apply(lambda x :x.skew()).sort_values(ascending=True)
    st.text(skew_features)
st.set_option('deprecation.showPyplotGlobalUse', False)
with st.echo():
    sns.displot(ns_ds, kind="kde", bw_adjust=.25)
    st.pyplot()

with st.echo():
    plt.figure(figsize=(20,20))
    plt.subplot(3,3,1)
    sns.boxplot(ns_ds['close'])
    plt.subplot(3,3,2)
    sns.boxplot(ns_ds['high'])
    plt.subplot(3,3,3)
    sns.boxplot(ns_ds['adjClose'])
    plt.subplot(3,3,4)
    sns.boxplot(ns_ds['adjHigh'])
    plt.subplot(3,3,5)
    sns.boxplot(ns_ds['adjVolume'])
    plt.subplot(3,3,6)
    sns.boxplot(ns_ds['divCash'])
    plt.subplot(3,3,7)
    sns.boxplot(ns_ds['splitFactor'])
    st.pyplot()

st.write("""
### Removing Skewness using Log Function
""")
with st.echo():
    out_cols_log = np.log(ns_ds)
    st.dataframe(out_cols_log)

with st.echo():
    st.text(out_cols_log.skew())

with st.echo():
    sns.displot(out_cols_log)
    st.pyplot()

st.write("""
### Removing Skewness using IQR
""")

with st.echo():
    Q1 = ns_ds.quantile(0.25)
    Q3 = ns_ds.quantile(0.75)
    IQR = Q3 - Q1
    t = ns_ds[~((ns_ds< (Q1 - 1.5 * IQR)) |(ns_ds > (Q3 + 1.5 * IQR))).any(axis=1)]
    st.text(t.skew())

with st.echo():
    sns.displot(t)
    st.pyplot()

with st.echo():
    plt.figure(figsize=(20,20))
    plt.subplot(3,3,1)
    sns.boxplot(t['close'])
    plt.subplot(3,3,2)
    sns.boxplot(t['high'])
    plt.subplot(3,3,3)
    sns.boxplot(t['adjClose'])
    plt.subplot(3,3,4)
    sns.boxplot(t['adjHigh'])
    plt.subplot(3,3,5)
    sns.boxplot(t['adjVolume'])
    plt.subplot(3,3,6)
    sns.boxplot(t['divCash'])
    plt.subplot(3,3,7)
    sns.boxplot(t['splitFactor'])
    st.pyplot()

st.write("""
*** HENCE WE OBSERVE THAT ALMOST ALL THE OUTLIERS OF THE DATA HAVE BEEN REMOVED ***
*** We observe that compared to Log method IQR method is more efficient ***
""")

with st.echo():
    st.dataframe(t)

st.write("""
## EDA
""")

with st.echo():
    import random
    st.write(df['close'].sample(5).mean())

with st.echo():
    st.write(df['close'].sample(5).median())

with st.echo():
    st.text(df['close'].sample(5).mode())

with st.echo():
    df_mean=df['close'].mean()
    st.write(df_mean)

with st.echo():
    df_median=df['close'].median()
    st.write(df_median)

with st.echo():
    df_mode=df['close'].mode()
    st.write(df_mode)

with st.echo():
    st.write(df.shape)

with st.echo():
    sum=0
    i=int(0)
    for i in range(0,1257):
        sum=sum+pow((df['close'].loc[i]- df_mean),2)
    sum=sum/df.shape[0] 
    st.write(sum)
    stddev=pow(sum,0.5)
    st.write(stddev)

# with st.echo():
    # df_close=df[['close','open']].agg([np.mean, np.std])
    # st.dataframe(df_close.transpose())

# df_close.transpose().plot(kind = "barh", y = "mean", legend = False, title = "Open and Close means")

# df_close.transpose().plot(kind = "barh", y = "mean", legend = False, title = "Open and Close Standard deviations",xerr = "std")

with st.echo():
    variance=pow(stddev,2)
    st.write(variance)

st.write("""
### Sampling
""")

with st.echo():
    samplemeans=[]
    for i in range(100):
        samples= df['close'].sample(n=10)
        samplemean=np.mean(samples)
        samplemeans.append(samplemean)
    st.write(samplemeans)    
    st.write(np.mean(samplemeans))

with st.echo():
    samples.plot(kind = "bar", x = "samplemeans", legend = True,title = "sample means")
    st.pyplot()

with st.echo():
    
    n = st.number_input('Enter the value of n',value=3,step=1)
    def systematic_sampling(df, step):  
        indexes = np.arange(0, len(df), step=step)
        systematic_sample = df.iloc[indexes]
        return systematic_sample
    systematic_sample = systematic_sampling(df, n)
    st.dataframe(systematic_sample)


with st.echo():
    systematic_data = round(systematic_sample['close'].mean())
    st.write("Systematic sampling mean (for close)", systematic_data)    

st.write("""
### Target Prediction
""")

with st.echo():
    target=150
    fig, ax = plt.subplots(figsize=(12,8))
    sns.distplot(np.mean(df.close), kde=False, label='Close')
    ax.set_xlabel("Close Value",fontsize=16)
    ax.set_ylabel("Frequency",fontsize=16)
    plt.axvline(target, color='red')
    plt.legend()
    plt.tight_layout()
    st.pyplot()

st.write("""
## Hypothesis Testing    
""")

with st.echo():
    apple_stationarity=t[['adjClose']]
    apple_stationarity.plot()
    st.pyplot()

st.write("""
From the plotted graph we can say that the data doesn't have a constant average as there are leaps and troughs and also the variance is also different at different stages of the data.
So our data is not stationary. We can also mathematically test for stationarity with adfuller test.

1.Augmented Dickey Fuller Test
The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.The intuition behind a unit root test is that it determines how strongly a time series is defined by a trend.
There are a number of unit root tests and the Augmented Dickey-Fuller may be one of the more widely used.

We interpret this result using the p-value from the test. 
A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).

p-value > 0.05: Not to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
""")

with st.echo():
    #Ho: Data is non stationary
    #H1: Data is stationary
    def adfuller_test(price):
        result=adfuller(price)
        labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
        for value,label in zip(result,labels):
            st.write(label,' : ',str(value))
        if result[1] <= 0.05:
            st.write("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


with st.echo():
    st.write(adfuller_test(apple_stationarity['adjClose']))

st.write("""
Since our p value is greater than 0.05 we need to accept the null hypothesis which states that our data is non-stationary.

Stationarity Conversion with shift()
Now let's convert our non-stationary data to stationary with shift() method. Here we take a shift() of 1 day which means all the records will step down to one step and we take the difference from the original data.
Since we see a trend in our data, when we subtract today's value from yesterday's value considering a trend it will leave a constant value on its way thus making the plot stationary.
""")

with st.echo():
    apple_stationarity['Close First Difference']=apple_stationarity['adjClose']-apple_stationarity['adjClose'].shift(1)
    apple_stationarity['Close First Difference'].plot()
    st.pyplot()


with st.echo():
    apple_stationarity=apple_stationarity['Close First Difference'].dropna()
    st.text(adfuller_test(apple_stationarity))

st.write("""
2. Z - test
It is believed that a stock price for a Apple company will grow at a rate of 200 USD per week with a standard deviation of 15 USD. An investor believes the stock won’t grow as quickly.
The changes in stock price is recorded for ten weeks and are as follows: $40, $30, $20, $32, $15, $7, $25, $12, $15, $22. 
Perform a hypothesis test using a 5% level of significance. State the null and alternative hypotheses, state your conclusion, and identify the Type I errors.
""")

with st.echo():
    #H0 : μ = 200   H1 : μ < 200 
    d=pd.Series([40, 30,20,32,15,7,25,12,15,22])
    n = 10
    xbar = d.mean()
    mu = 200
    sigma = 15  
    alpha = 0.05
    z = (xbar - mu)/(sigma/np.sqrt(n))
    p = (stats.norm.cdf(z))
    if(p<=alpha):
        rejection = 'Reject Null Hypothesis'
    else:
        rejection = 'Accept Null Hypothesis'
    st.write(rejection,round(z,2),p)

st.write("""
p < 0.05 so we reject the null hypothesis.
There is sufficient evidence to suggest that the stock price of the company grows at a rate less than 200 USD a week.

Type I Error: To conclude that the stock price is growing slower than 200 USD a week when, in fact, the stock price is growing at 200 USD a week (reject the null hypothesis when the null hypothesis is true).
Type II Error: To conclude that the stock price is growing at a rate of 200 USD a week when, in fact, the stock price is growing slower than 200 USD a week (do not reject the null hypothesis when the null hypothesis is false).


# Z-Test using P-value
The average closing price of the Apple stocks is 188.7 USD.Investors were interested in seeing if supply and demand affects the closing price.
50 days of close price over a period of randomly selected weeks yielded a sample mean of 199.2 feet. The population standard deviation is known to be 70.4.
Can it be concluded at the 0.05 level of significance that the average closing price has increased? Is there evidence of what caused this to happen?

""")

with st.echo():
    #H0: μ=188.7
    #H1: μ>188.7
    #Initial variable here
    n = 50
    xbar=199.2
    mu = 188.7
    sigma = 70.4
    alpha = 0.05
    z = (xbar - mu)/(sigma/np.sqrt(n))
    z_critical = abs(stats.norm.ppf(0.05))
    if(abs(z) < z_critical): 
        rejection = 'Accept Null Hypothesis'
    else:
        rejection = 'Reject Null Hypothesis'
    st.write(rejection, z, z_critical)

st.write("""
ztest < zcritcal so we do not reject the null hypothesis.
Thus there is no sufficient evidence to suggest that the average closing price has increased.
""")

with st.echo():
    import pandas as pd
    import numpy as np

    # reading the data

    # looking at the first five rows of the data
    print(df.head())
    print('\n Shape of the data:')
    print(df.shape)

    # setting the index as date
    df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
    df.index = df['date']

    #creating dataframe with date and the target variable
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['date', 'close'])

    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]

    # NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

    # splitting into train and validation
    train = new_data[:987]
    valid = new_data[987:]

    # shapes of training set
    print('\n Shape of training set:')
    print(train.shape)

    # shapes of validation set
    print('\n Shape of validation set:')
    print(valid.shape)

    # In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
    # making predictions
    preds = []
    Sum=0
    for i in range(0,valid.shape[0]):
        a = train['close'][len(train)-248+i:].sum() + Sum
        b = a/248
        Sum+=b
        preds.append(b)

    # checking the results (RMSE value)
    rms=np.sqrt(np.mean(np.power((np.array(valid['close'])-preds),2)))
    print('\n RMSE value on validation set:')
    print(rms)

with st.echo():
    valid['Predictions'] = 0
    valid['Predictions'] = preds
    plt.plot(valid[['close', 'Predictions']])
    plt.plot(train['close'])
    st.pyplot()

with st.echo():
    #importing required libraries
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    #creating dataframe
    data = df2.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df2)),columns=['date', 'close'])
    for i in range(0,len(data)):
        new_data['date'][i] = data['date'][i]
        new_data['close'][i] = data['close'][i]

    #setting index
    new_data.index = new_data.date
    new_data.drop('date', axis=1, inplace=True)

    #creating train and test sets
    dataset = new_data.values

    train = dataset[0:987,:]
    valid = dataset[987:,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

with st.echo():
    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
    st.write(rms)

with st.echo():
    #for plotting
    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price
    plt.plot(train['close'])
    plt.plot(valid[['close','Predictions']])
    st.pyplot()

with st.echo():
    from sklearn.linear_model import LinearRegression 
    #   pandas and numpy are used for data manipulation 
    import pandas as pd 
    import numpy as np 
    # matplotlib and seaborn are used for plotting graphs 
    import matplotlib.pyplot as plt 
    import seaborn 
    # fix_yahoo_finance is used to fetch data 
    #import fix_yahoo_finance as yf
    #Df = yf.download('AAPL','2008-01-01','2020-12-31')
    # Only keep close columns 
    df2=df2[['close']] 
    # Drop rows with missing values 
    df2= df2.dropna() 
    # Plot the closing price of GLD 
    df2.close.plot(figsize=(10,5)) 
    plt.ylabel("AAPL Prices")
    plt.show()
    st.pyplot()

with st.echo():
    df2['S_3'] = df2['close'].shift(1).rolling(window=3).mean() #Moving Avg
    df2['S_9']= df2['close'].shift(1).rolling(window=9).mean() 
    df2= df2.dropna() 
    X = df2[['S_3','S_9']] 
    X.head()
    y = df2['close']
    y.head()

with st.echo():
    t=.7 
    t=int(t*len(df2)) 
    # Train dataset 
    X_train = X[:t] 
    y_train = y[:t]  
    # Test dataset 
    X_test = X[t:] 
    y_test = y[t:]

with st.echo():
    linear = LinearRegression().fit(X_train,y_train)
    predicted_price = linear.predict(X_test)  
    predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
    predicted_price.plot(figsize=(10,5))  
    y_test.plot()  
    plt.legend(['predicted_price','actual_price'])  
    plt.ylabel("AAPL Price")  
    plt.show()
    st.pyplot()

with st.echo():
    r2_score = linear.score(X[t:],y[t:])
    print("Accuracy :",float("{0:.2f}".format(r2_score*100)))