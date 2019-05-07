from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# 最多使用的单词数。
max_features = 20000
maxlen = 80  
batch_size = 32

# 加载数据并将单词转化为ID，max_features给出了最多使用的单词数。
(trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
print(len(trainX), 'train sequences')
print(len(testX), 'test sequences')

# 在自然语言中，每一段话的长度是不一样的，但循环神经网络的循环长度是固定的，
# 所以这里需要先将所有段落统一成固定长度。
trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
testX = sequence.pad_sequences(testX, maxlen=maxlen)
print('trainX shape:', trainX.shape)
print('testX shape:', testX.shape)

# 构建模型
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=10,
          validation_data=(testX, testY))

score = model.evaluate(testX, testY, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
