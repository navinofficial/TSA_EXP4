# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 16/5/26
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('gold_rate_history.csv')
df.head()

df.drop(['Country', 'State', 'Location'], axis=1, inplace=True)
df.head()

df.columns = df.columns.str.strip()
df.columns
df['rate'] = df['rate'].replace('[\$,]', '', regex=True).astype(float)

df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df = df.sort_values(by='Date')
df.set_index('Date', inplace=True)

N=1000
plt.rcParams['figure.figsize'] = [12, 6]
X=df['rate']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()
```
## SIMULATED ARMA(1,1) PROCESS:
```
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
```
## Partial Autocorrelation
```
plot_pacf(ARMA_1)
plt.show()
```
## Autocorrelation

```
plot_acf(ARMA_1)
plt.show()
```
## SIMULATED ARMA(2,2) PROCESS:
```
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
```
## Partial Autocorrelation

```
plot_pacf(ARMA_2)
plt.show()
```

## Autocorrelation
```
plot_acf(ARMA_2)
plt.show()
```


OUTPUT:
## Original Data:
<img width="821" height="431" alt="image" src="https://github.com/user-attachments/assets/aeaca70a-f6a5-49b0-ac72-8fe9192a4759" />
<img width="994" height="241" alt="image" src="https://github.com/user-attachments/assets/db97ec0e-0541-4966-83e9-50b10c036b39" />
<img width="993" height="241" alt="image" src="https://github.com/user-attachments/assets/c4486aa0-8331-449d-95a4-2554cae96ebf" />

## SIMULATED ARMA(1,1) PROCESS:
<img width="832" height="438" alt="image" src="https://github.com/user-attachments/assets/97740145-6a42-4a19-88f7-bab178e30066" />
## Partial Autocorrelation
<img width="825" height="435" alt="image" src="https://github.com/user-attachments/assets/e108679b-0217-46fd-9336-566845043c8a" />

## Autocorrelation
<img width="830" height="437" alt="image" src="https://github.com/user-attachments/assets/d212a63b-4786-4b66-853e-b8e8caccf4dc" />

## SIMULATED ARMA(2,2) PROCESS:
<img width="823" height="439" alt="image" src="https://github.com/user-attachments/assets/678b2c5e-0df4-4a33-8345-e31a9b89eae4" />

## Partial Autocorrelation
<img width="826" height="438" alt="image" src="https://github.com/user-attachments/assets/05c37ae0-91dd-446a-b85f-3229b75e0592" />

## Autocorrelation
<img width="828" height="435" alt="image" src="https://github.com/user-attachments/assets/6943c81f-217d-4e85-a833-2777ce16bc20" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
