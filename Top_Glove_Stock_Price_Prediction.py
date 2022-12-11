# %%
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras import Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import numpy as np
import os, datetime

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

# %% 1. Data loading
df = pd.read_csv(os.path.join(DATASET_PATH, 'Top_Glove_Stock_Price_Train(Modified).csv'))

# %% 2. Data inspection
df.head(10)
df.tail(10)

df.info()
df.describe()

df.isna().sum() # 5 NaN value in Open column
df.duplicated().sum() # No duplicated values

# Visualization
# Plot missing data
msno.matrix(df)

# Plot Open column to see data trend and how missing values influence the trend
plt.figure(figsize=(10,10))
plt.plot(df['Open'])
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Open Stock Price Through Time')
plt.show()

# %% 3. Data cleaning
# Fill missing values in the Open column with spline interpolation
df['Open'] = df['Open'].interpolate(method='spline', order=2)
df['Open'].isna().sum()

plt.figure(figsize=(10,10))
plt.plot(df['Open'])
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Open Stock Price Through Time')
plt.show()

# %% 4. Feature selection
# Define the features that will be used for the prediction
data = df['Open']

# %% 5. Data pre-processing
# Expand the dimension of the array from 1d to 2d
data = np.reshape(data.values, (-1,1))

# Using MinMaxScaler to normalize data
mm_scaler = MinMaxScaler()
data = mm_scaler.fit_transform(data)

# Split the data according to a set window size
WINDOW_SIZE = 60
SEED = 12345
X_data = []
Y_data = []

for i in range(WINDOW_SIZE, len(data)):
    X_data.append(data[i-WINDOW_SIZE:i])
    Y_data.append(data[i])

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Do train-validation split on the data
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, random_state=SEED)

# %% 6. Model development
# Define the input size
input_size = X_train.shape[1:]

# Define the model and its layers
model = Sequential()
model.add(LSTM(64, input_shape=input_size))
model.add(Dense(1))

# Print the model summary
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

# %%
# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Define the callbacks
LOG_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_PATH)
es = EarlyStopping(patience=10, restore_best_weights=True)

# Train the model
EPOCHS = 100
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=EPOCHS, callbacks=[tb, es])

# %% 7. Model evaluation
# Load the test data
feature_names = df.columns
test_df = pd.read_csv(os.path.join(DATASET_PATH, 'Top_Glove_Stock_Price_Test.csv'), names=feature_names)

# Concatenate the Open column in the test and train data
test_data = pd.concat((df['Open'], test_df['Open']))
test_data = test_data.iloc[len(test_data)-WINDOW_SIZE-len(test_df):]

# Expand the dimension of the test data
test_data = np.expand_dims(test_data, -1)

# Do normalization on the test data
test_data = mm_scaler.transform(test_data)

# Cut the test data according to the window size
X_eval = []
Y_eval = []

for i in range(WINDOW_SIZE, len(test_data)):
    X_eval.append(test_data[i-WINDOW_SIZE:i])
    Y_eval.append(test_data[i])

X_eval = np.array(X_eval)
Y_eval = np.array(Y_eval)

# Do prediction with the model
prediction = model.predict(X_eval)

# Using the scaler to inverse the transformation
Y_eval = mm_scaler.inverse_transform(Y_eval)
prediction = mm_scaler.inverse_transform(prediction)

# Evaluate the prediction
mae = mean_absolute_error(Y_eval, prediction)
mape = mean_absolute_percentage_error(Y_eval, prediction)

# Plot a graph of the prediction againts the target
plt.figure()
plt.plot(prediction, color='red', label='Prediction')
plt.plot(Y_eval, color='blue', label='Stock Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction Againts Real Price')
plt.show()

print('MAE: {}\nMAPE: {}'.format(mae, mape))
