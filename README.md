# Geopolitical Risk Analysis and its Impact on Market Volatility

The aim is to develop a predictive model that analyses the impact of geopolitical risks on market volatility i.e., post-COVID era, post Russia-Ukraine war era etc. By examining historical financial data and the Geopolitical Risk Index (GPRI), the model will forecast market behavior in response to geopolitical events. This analysis will enable investors and analysts to differentiate between typical market movements and those triggered by geopolitical unrest, guiding informed financial decisions in volatile times and enabling long-term understanding.
My approach has involved aggregating a comprehensive dataset encompassing key financial indices and the Geopolitical Risk Index (GPRI) to serve as an indicator for geopolitical instability. We have employed advanced machine learning approaches, including Bi-LSTM networks augmented with attention mechanisms, to capture both the temporal dependencies and the nuanced impact of critical events on market volatility. The model has been meticulously trained and tested on historical data, with particular attention to the period marked by the COVID-19 pandemic.

### Stock Indices Used

1. S&P 500 (America)
2. GDAXI (Germany)
3. FCHI (France)
4. FTSE (Britain)
5. IOMEX (Russia)
6. NIFTY (India)
7. SSE (China)
8. GPRI (Geopolitical Risk Index)

### Bi-LSTM

| Layer | Output Shape | Param # |
| --- | --- | --- |
| input_4 (InputLayer) | (None, 30, 6) | 0 |
| bidirectional_3 | (None, 100) | 22,800 |
| dense_6 | (None, 50) | 5,050 |
| dropout_3 | (None, 50) | 0 |
| dense_7 | (None, 1) | 51 |

### Bi-LSTM with Attention

| Layer | Output Shape | Param # |
| --- | --- | --- |
| input_4 (InputLayer) | [(None, 30, 6)] | 0 |
| bidirectional_3 (Bidirectional) | (None, 30, 100) | 22,800 |
| attention_3 (Attention) | (None, 30, 100) | 130 |
| flatten_3 (Flatten) | (None, 3000) | 0 |
| dense_6 (Dense) | (None, 50) | 150,050 |
| dropout_3 (Dropout) | (None, 50) | 0 |
| dense_7 (Dense) | (None, 1) | 51 |

**************************************The Attention Layer**************************************

- The attention mechanism is advantageous for predicting volatile sequences because it can highlight the impact of specific geopolitical events.
- Adding an attention layer to the Bi-LSTM model improves its performance on datasets with pronounced events or changes by allowing it to focus on relevant parts of the sequence. This is in contrast to the initial Bi-LSTM model, which treats all parts of the sequence equally and may not capture critical temporal features.

### Complex Bi-LSTM with Attention and Optimization

| Layer | Output Shape | Param # |
| --- | --- | --- |
| input_20 (InputLayer) | [(None, 30, 6)] | 0 |
| bidirectional_21 (Bidirectional) | (None, 30, 200) | 85,600 |
| bidirectional_22 (Bidirectional) | (None, 100) | 100,400 |
| dense_38 (Dense) | (None, 50) | 5,050 |
| dropout_19 (Dropout) | (None, 50) | 0 |
| dense_39 (Dense) | (None, 1) | 51 |

**Bi-LSTM Layers**: Two bidirectional LSTM layers are used to process the input data. The first LSTM layer has 100 units and returns sequences to feed into the next LSTM layer. 

**Custom Loss Function**: A weighted mean squared error (MSE) loss function is defined, which assigns larger weights to larger errors. This means that predictions that are far from the actual values will contribute more to the loss, encouraging the model to focus on minimizing larger errors.

**Learning Rate Scheduler**: A custom callback function **`scheduler`** is used to adjust the learning rate during training. For the first 10 epochs, it maintains the initial learning rate, and after that, it exponentially decreases the learning rate. This approach is often used to fine-tune the model as training progresses, allowing for larger updates initially and smaller, more precise updates later in training.

## Bi-LSTM Architecture

### Results

<img width="859" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/45075c75-fd9a-4eef-aee1-b3a5b3157eb4">


| Metric | Value |
| --- | --- |
| Mean Absolute Error | 0.11900745609308362 |
| Root Mean Squared Error | 0.21580776031679524 |
| Mean Squared Error | 0.04657298941295134 |

### Inference

1. **Learning Trends**: The model is learning as indicated by the decreasing loss on the training set. It starts with a higher loss and then stabilises as training progresses.
2. **Validation Loss Higher Than Training Loss**: The validation loss doesn't increase as training progresses, which is a good sign that the model isn't overfitting significantly.
3. **Small Gap Between Training and Validation Loss**: The small gap between the training and validation loss suggests that the model generalises well. There isn't a large discrepancy between performance on the training data and unseen validation data.
4. **Stability Over Epochs**: Both the training and validation loss seem to plateau, indicating that further training might not yield significant improvements without changes to the model or training procedure. This could suggest that the model has reached its capacity given the current architecture and data. Early stopping could also be implemented.

<img width="1009" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/c2980342-2817-43c8-8209-6ddcc0835963">


### Inference

1. **General Trend**: The model seems to capture the general trend of the actual data, as indicated by the blue predicted line sometimes following the red actual line. However, there are noticeable deviations.
2. **Predictive Performance**: The model has predictive capability, as indicated by the sections where the predicted values closely follow the actual values. However, it struggles with rapid changes or outliers in the data.
3. **Potential Over-Smoothing**: The model's predictions are smoother compared to the actual values, which might indicate the model is averaging out the input data too much, a common issue with many time series forecasting models.

<img width="846" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/e810363c-83ec-4623-a983-12dc6537b1c0">


### Inference

1. **Centered Around Zero**: A good sign. This indicates that the model is not systematically over-predicting or under-predicting.
2. **No Clear Pattern**: This suggests that the model is capturing the variance in the data appropriately.
3. **Homoscedasticity**: The spread of the residuals should be consistent across the x-axis (time). The model does not struggle with data points away from the mean.

<img width="848" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/14f2c040-d252-4297-9c31-dcf9c1592a89">


**Confidence Intervals**: The shaded gray area around the predicted line represents the confidence intervals. This interval shows the range within which we expect the actual values to lie, given a certain level of confidence (typically 95%). The width of the shaded area indicates the level of uncertainty in the predictions; a wider area suggests more uncertainty.

## Bi-LSTM with Attention Layer

<img width="1008" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/db8f0c87-5e9f-41b1-91c1-6536180f7031">


| Metric | Value |
| --- | --- |
| Mean Absolute Error | 0.13173182586215845 |
| Root Mean Squared Error | 0.21971236652685194 |

The model's predictions are less smoother and more responsive to sharp peaks and troughs in the actual stock price movements. This might suggest the model is not averaging the input data over the sequence, preserving detail on rapid market changes.

## Complex Bi-LSTM with Attention Layer

<img width="1159" alt="image" src="https://github.com/keshavrathinavel/Geopolitical-Risk-Analysis-and-Its-Impact-on-Market-Volatility/assets/73035121/c039fe94-004e-48a5-a2df-ea6324710854">


### Inferences

1. **Trend Following**: The model's predicted GPRI (blue line) appears to follow the overall trend of the actual GPRI (red line), particularly evident in the broader movements over time. This suggests that the model has successfully learned the general direction and trend of the GPRI.
2. **Baseline Predictions**: The model establishes a baseline prediction that seems to capture the non-volatile or less erratic periods of the GPRI, indicating it has learned the underlying pattern when the index is stable.
3. **Reactivity to Changes**: Although the model does not capture the spikes, it reacts to changes in the actual GPRI, as seen by the blue line's shifts corresponding to movements in the red line, albeit with a delay and less intensity.
4. **Smoothness**: The smooth nature of the predicted GPRI indicates the model’s ability to filter out noise from the data, focusing on the underlying factors driving the GPRI movements, which could be beneficial for certain types of risk assessment or long-term planning.

# Conclusion

Throughout this research, we have developed and evaluated a predictive model focused on the Geopolitical Risk Index (GPRI), an essential metric for understanding the interplay between geopolitical events and financial market volatility. Our approach, leveraging a sophisticated Bi-LSTM network with an attention mechanism, has demonstrated a promising ability to capture the overarching trends of the GPRI.

The model's performance, particularly in following the general direction of the GPRI, underscores its potential in trend analysis and long-term market assessments. By smoothing out short-term fluctuations, the model provides a filtered perspective on the GPRI, which can be highly valuable for strategic planning and risk assessment purposes. The adaptability of the model to learn from historical data and predict future trends suggests that it has successfully internalised key temporal patterns that are indicative of broader market behaviors in response to geopolitical risks.

Moreover, the model's avoidance of overreaction to market noise and its focus on sustained trends is a testament to its robustness. This characteristic is particularly advantageous in creating stable, long-term financial strategies that can withstand the test of volatile market conditions.
