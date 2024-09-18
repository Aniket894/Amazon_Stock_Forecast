# Amazon Stock Forecasting Project Documentation

This project aims to forecast Amazon stock prices between `2023-07-01` and `2024-07-15`, utilizing various time series models, machine learning models, and deep learning techniques. The dataset is composed primarily of the closing stock prices (`close` column), although multivariate models incorporate additional columns like `open`, `high`, `low`, and `volume`. The project is divided into four notebooks, each focusing on different forecasting methods, including ARIMA, SARIMA, machine learning models, and LSTM (Long Short-Term Memory) models.

### Data Overview
The stock data is sourced from public financial datasets and contains features such as:
- **Date**: The trading date.
- **Open**: Opening price of Amazon stock for the given day.
- **Close**: Closing price of Amazon stock for the given day.
- **High**: Highest price during the trading day.
- **Low**: Lowest price during the trading day.
- **Volume**: Number of shares traded during the day.

The forecasting is done for two horizons:
1. **Short-term forecast**: Next 30 days.
2. **Medium-term forecast**: Next 90 days.

---

## 1. Amazon Univariate Stock Forecast with ARIMA and SARIMA (`Amazon_Stock_forecast_(arima__and_sarima).ipynb`)

### Objective:
Forecast the closing price of Amazon stock using traditional univariate time series models (ARIMA and SARIMA).

### Steps:

#### 1.1. **Data Preprocessing**:
   - **Data Loading**: The dataset is loaded, and only the `close` column is used for forecasting.
   - **Time Series Decomposition**: The `close` column is decomposed into trend, seasonality, and residuals to visualize the underlying patterns and ensure stationarity.
   - **Stationarity Check**: The Augmented Dickey-Fuller (ADF) test is performed to determine if the time series is stationary.
   - **Differencing**: If the data is non-stationary, differencing is applied to make the series stationary, a prerequisite for ARIMA/SARIMA modeling.

#### 1.2. **ARIMA Model**:
   - **Model Identification**: Using Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots, appropriate values for ARIMA hyperparameters (p, d, q) are identified.
   - **Model Fitting**: The ARIMA model is trained on the `close` column data.
   - **Forecasting**:
     - Short-term: 30-day forecast.
     - Medium-term: 90-day forecast.
   - **Performance Evaluation**: Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are calculated on test data, and the results are visualized.

   - Next 30 Days Forecast

   - ![download (66)](https://github.com/user-attachments/assets/7715c2ee-62a9-4704-a4cc-338cfb1af49e)

   - Next 90 Days Forecasat

   - ![download (70)](https://github.com/user-attachments/assets/bd87bca0-6d97-40d1-bdba-88baa42acea9)



#### 1.3. **SARIMA Model**:
   - **Model Identification**: Seasonal ARIMA (SARIMA) hyperparameters (p, d, q, P, D, Q, m) are chosen based on seasonality in the data.
   - **Model Fitting**: The SARIMA model is fitted to the `close` column.
   - **Forecasting**:
     - Short-term: 30-day forecast.
     - Medium-term: 90-day forecast.
   - **Performance Evaluation**: Similar metrics (MSE, RMSE) are used to evaluate the SARIMA model, with a focus on improved accuracy over ARIMA due to seasonality handling.

   - Next 30 Days Forecast

   - ![download (67)](https://github.com/user-attachments/assets/e3d9a06e-2068-4e1b-af8c-93a4e81a77c0)

   - Next 90 Days Forecast

   - ![download (71)](https://github.com/user-attachments/assets/076ecfa7-4d7b-46be-a3fc-bd377b090105)



#### 1.4. **Visualization**:
   - The actual vs forecasted values are plotted for both 30-day and 90-day horizons, with a comparison of ARIMA and SARIMA performance.

   - Next 30 Days Forecast with arima and sarima

   - ![download (68)](https://github.com/user-attachments/assets/6afe5c90-323a-49b1-8401-09e2c00d237f)

   - Next 90 Days Forecast with arima and sarima

   - ![download (72)](https://github.com/user-attachments/assets/f09e01c4-8aaa-45ec-971a-5922138efd0a)



---

## 2. Amazon Univariate Stock Forecast with Machine Learning Models (`Amazon_Stock_forecastin_with_ML_Models(1).ipynb`)

### Objective:
Forecast the closing price of Amazon stock using traditional machine learning models such as Linear Regression, Random Forest, Support Vector Regression (SVR), and XGBoost (XGB) Regressor.

### Steps:

#### 2.1. **Data Preprocessing**:
   - **Data Loading**: The `close` column is loaded, and the data is split into training and testing sets.
   - **Feature Scaling**: The `close` prices are scaled using MinMaxScaler or StandardScaler to improve the performance of machine learning algorithms that are sensitive to the scale of input data.

#### 2.2. **Model Training and Forecasting**:
   - **Linear Regression**:
     - A simple linear regression model is trained on the `close` column.
     - Forecasts are made for the next 30 and 90 days.
    
     - Next 30 Days Forecast
    
     - ![download (74)](https://github.com/user-attachments/assets/99e61868-f912-4063-a64d-1e8696ca232a)
  
     - ![download (76)](https://github.com/user-attachments/assets/4cc3d888-c600-4290-beed-b9137d50f375)
    
     - Next 90 Days Forecast
       
     - ![download (77)](https://github.com/user-attachments/assets/3fabfcc0-5bf8-4a01-93dc-9db476ab775f)
    
     - ![download (78)](https://github.com/user-attachments/assets/80954896-5035-4789-9658-fad89da3ad0b)




   
   - **Random Forest Regressor**:
     - A Random Forest model is trained on the `close` column.
     - Forecasts are generated for 30 and 90 days.
    
     - Next 30 Days Forecast
    
     - ![download (79)](https://github.com/user-attachments/assets/bda5620b-d60a-46e4-8667-52b189a43146)
    
     - ![download (80)](https://github.com/user-attachments/assets/7f642e3b-b193-4d50-a2fd-c6b956ed5a81)
    
     - Next 90 Days Forecast
    
     - ![download (81)](https://github.com/user-attachments/assets/14605cb7-04b7-46e1-a91d-d6c7eeb03331)
    
     - ![download (82)](https://github.com/user-attachments/assets/c60d73cf-39dd-4e53-acc2-2a0f2adfdd0b)




   
   - **Support Vector Regression (SVR)**:
     - A Support Vector Machine-based regression model is used to predict the closing price.
     - Forecasts for 30 days and 90 days are generated.
    
     - Next 30 Days Forecast
    
     - ![download (83)](https://github.com/user-attachments/assets/4d088f91-ddc4-4a1c-b068-5aa976e3590b)
       
    
     - Next 90 Days Forecast
    
     - ![download (84)](https://github.com/user-attachments/assets/6814c51f-c63d-4834-b264-86b7a538ada6)

    
   
   - **XGBoost Regressor (XGB)**:
     - The XGBoost regressor is applied to predict the stock prices for the next 30 and 90 days.
    
     - Next 30 Days Forecast
    
     - ![download (85)](https://github.com/user-attachments/assets/47b793ef-c14c-4ae9-9246-81baec00b8fc)
     - 
    
     - Next 90 Days Forecast
    
     - ![download (86)](https://github.com/user-attachments/assets/301016e9-924d-453e-9306-497c9d6181f5)



#### 2.3. **Performance Evaluation**:
   - The models are evaluated using metrics like MSE, RMSE, and R-squared to measure their forecasting accuracy.
   - A comparison is made across the machine learning models, highlighting which model performs better for the given dataset.

---

## 3. Amazon Univariate Stock Forecast with LSTM (`Amazon_Univariate_Stock_forecast_with_LSTM.ipynb`)

### Objective:
Use a univariate Long Short-Term Memory (LSTM) neural network to forecast Amazon’s closing stock prices.

### Steps:

#### 3.1. **Data Preprocessing**:
   - **Data Loading**: Only the `close` column is loaded.
   - **Feature Scaling**: MinMaxScaler is applied to normalize the closing prices between 0 and 1.
   - **Data Reshaping**: The time series data is reshaped to fit the LSTM input format, which expects sequences (timesteps) as input.
   - **Train-Test Split**: Data is split into training and test sets.

#### 3.2. **LSTM Model Building**:
   - **Model Architecture**:
     - Input layer with the number of time steps.
     - One or more LSTM layers with a specified number of units.
     - Dropout layers to prevent overfitting.
     - Dense output layer to forecast the closing price.
   
   - **Model Training**:
     - The LSTM model is trained on the scaled closing prices using a sequence of past prices to predict future prices.
   
   - **Forecasting**:
     - The model is used to forecast the closing price for the next 30 days and 90 days.

#### 3.3. **Performance Evaluation**:
   - The model is evaluated using MSE, RMSE, and MAE, comparing the predicted values against the actual test data.

#### 3.4. **Visualization**:
   - Forecasted vs actual closing prices are plotted to visualize the model’s performance over the forecasting horizons.

   - Next 30 Days Forecast

   - ![download (87)](https://github.com/user-attachments/assets/aba04ec1-3dde-4dbc-a90b-0542b408be11)

   - ![download (88)](https://github.com/user-attachments/assets/60b2b571-bccc-419b-925f-a209a30d8322)

   - Next 90 Days Forecast

   - ![download (89)](https://github.com/user-attachments/assets/5b3641f8-f000-4e8a-b4e6-7f32212e81a5)

   - ![download (90)](https://github.com/user-attachments/assets/c815570a-5653-4373-80a1-db792441feef)


---

## 4. Amazon Multivariate Stock Forecast with LSTM (`Amazon_Multivariate_Stock_forecast_with_LSTM.ipynb`)

### Objective:
Use a multivariate LSTM model to forecast Amazon’s stock price, leveraging multiple columns (e.g., open, high, low, close, volume) for improved accuracy.

### Steps:

#### 4.1. **Data Preprocessing**:
   - **Data Loading**: Multiple columns (`open`, `high`, `low`, `close`, `volume`) are loaded for the multivariate analysis.
   - **Feature Scaling**: All relevant columns are normalized using MinMaxScaler to ensure compatibility with the LSTM model.
   - **Data Reshaping**: The data is reshaped into sequences to meet LSTM input requirements.
   - **Train-Test Split**: The data is split into training and test sets, with sequences and multiple features as inputs.

#### 4.2. **Multivariate LSTM Model Building**:
   - **Model Architecture**:
     - Input layer that accommodates multiple time steps and multiple features.
     - LSTM layers with a specified number of units.
     - Dense layer for output to predict the `close` price.
   
   - **Model Training**:
     - The LSTM model is trained on multiple features to predict the `close` column.
   
   - **Forecasting**:
     - The multivariate LSTM model is used to forecast the stock prices for the next 30 and 90 days.

#### 4.3. **Performance Evaluation**:
   - Evaluation metrics like MSE, RMSE, and MAE are used to assess the model’s predictive accuracy on the test data.
   - Comparison with univariate LSTM results to assess if the multivariate model improves forecasting accuracy.

#### 4.4. **Visualization**:
   - Forecasted vs actual stock prices are plotted, showcasing the model’s ability to predict future trends using multiple features.

   - Next 30 Days Forecast

   - ![download (91)](https://github.com/user-attachments/assets/f96e517e-2aa1-4715-8bfe-d2653bd45159)
     

   - Next 90 Days Forecast

   - ![download (92)](https://github.com/user-attachments/assets/1efc04a2-9eca-4d51-a10c-7bb113022dcf)



---

## Conclusion

The project explores different approaches to forecasting Amazon’s stock prices. From traditional time series models (ARIMA, SARIMA) to machine learning (Linear Regression, Random Forest, SVR, XGB) and deep learning (LSTM), each method is tested for short-term (30-day) and medium-term (90-day) forecasting horizons. The multivariate LSTM model provides a more comprehensive approach by incorporating multiple features, while the

 univariate models focus on the `close` column alone.

The comparison of these techniques provides insights into the strengths and weaknesses of each model, with evaluation metrics like MSE and RMSE used to determine the accuracy of the forecasts. This comprehensive analysis helps identify the most effective model for predicting future stock prices based on historical data.
