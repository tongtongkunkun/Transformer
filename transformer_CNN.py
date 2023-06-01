import numpy as np
import pandas as pd
import shap
from keras.callbacks import LearningRateScheduler
from keras.layers import MultiHeadAttention, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Layer
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Permute

# 读取CSV文件
df = pd.read_csv('Hazardous_waste_prediction5.csv')

# 划分输入和输出
X_or = df.iloc[:, 1:].values
y_or = df.iloc[:, 0].values

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X_or)
y = scaler.fit_transform(y_or.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将训练数据转化为Transformer模型的输入格式

# 设置时间步长
TIME_STEPS = 1

# 设置特征维度
FEATURES = X_train.shape[1]

# 构建训练集数据
X_train = X_train.reshape((X_train.shape[0], TIME_STEPS, FEATURES))
X_test = X_test.reshape((X_test.shape[0], TIME_STEPS, FEATURES))

y_train = y_train.reshape((y_train.shape[0], 1))
y_test = y_test.reshape((y_test.shape[0], 1))

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, dtype=tf.float32)

class PositionWiseFeedForward(Layer):
    def __init__(self, d_model, dff, activation='relu'):
        super(PositionWiseFeedForward, self).__init__()
        self.dense1 = Dense(dff, activation=activation)
        self.dense2 = Dense(d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dense1': self.dense1,
            'dense2': self.dense2
        })
        return config

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# 构建Transformer模型
def transformer_model(num_layers, d_model, num_heads, dff, input_shape, output_shape, key_dim, dropout_rate=0.1):
    inputs = Input(shape=input_shape)

    # 编码器
    x = Rescaling(scale=1.0 / d_model)(inputs)
    x = PositionalEncoding()(x)
    for i in range(num_layers):
        x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x,x,x)
        x = Dropout(rate=dropout_rate)(x)
        x = LayerNormalization(epsilon=1e-6)(inputs + x)
        x = PositionWiseFeedForward(d_model=d_model, dff=dff)(x)
        x = Dropout(rate=dropout_rate)(x)
        x = LayerNormalization(epsilon=1e-6)(inputs + x)

    # 解码器
    x = Permute((2, 1))(x)
    x = Conv1D(64, 2, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    outputs = x
    model = Model(inputs=inputs, outputs=outputs)

    return model


# 编译模型
model = transformer_model(num_layers=1, d_model=65, num_heads=8, dff=46, input_shape=(TIME_STEPS, FEATURES),
                          output_shape=(1,), key_dim=8)
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr = 2e-4
    if epoch > 20:
        lr = 1e-4
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)
model.compile(optimizer='adam', loss='mse')
print(model.summary())
log_dir = 'Multi_predict'
tensorboard_callback = TensorBoard(log_dir=log_dir)
# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1, validation_data=(X_test, y_test), callbacks=[lr_scheduler,tensorboard_callback])
# 在每个训练步骤结束时记录d_model的热力图


# 在测试集上评估模型性能
mse = model.evaluate(X_test, y_test, verbose=1)

from sklearn.metrics import mean_squared_error, r2_score

print(X_train.shape)

# 进行预测
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
print(trainPredict.shape)

# 反归一化
trainPredict = scaler.inverse_transform(trainPredict.reshape(trainPredict.shape[0], -1))
testPredict = scaler.inverse_transform(testPredict.reshape(testPredict.shape[0], -1))
trainY = scaler.inverse_transform(y_train)
testY = scaler.inverse_transform(y_test)
print(trainY, trainPredict)
print(testY, testPredict)

# 计算RMSE和R2
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train RMSE: %.3f' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test RMSE: %.3f' % (testScore))

trainR2 = r2_score(trainY, trainPredict)
print('Train R2: %.3f' % (trainR2))
testR2 = r2_score(testY, testPredict)
print('Test R2: %.3f' % (testR2))

# 绘制损失曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

df1 = pd.read_csv('Hazardous_waste_prediction8_average_future.csv')
# 划分输入和输出
X1 = df1.iloc[:, 1:].values
y1 = df1.iloc[:, 0].values

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)
y = scaler.fit_transform(y1.reshape(-1, 1))
# 将训练数据转化为Transformer模型的输入格式

# 设置时间步长
TIME_STEPS = 1

# 设置特征维度
FEATURES = X1.shape[1]

# 构建训练集数据
X1 = X1.reshape((X1.shape[0], TIME_STEPS, FEATURES))
y1 = model.predict(X1)
print(y1.shape)
future_Predict = scaler.inverse_transform(y1.reshape(y1.shape[0], -1))
print(future_Predict)




