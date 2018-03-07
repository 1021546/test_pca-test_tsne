#!/usr/bin/env python
import numpy
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
import wavio
import os
from keras.utils.vis_utils import plot_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

''' EarlyStopping '''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)

batch_size = 128
num_classes = 10
epochs = 10

def getTrainingData():
	temp=[]

	for k in range(1,11):
		for i in range(0,10):
			for j in range(1,6):
				filename="./wav/"+str(k)+"/"+str(i)+"_"+str(j)+".wav"
				# print(filename)
				w= wavio.read(filename)
				# print("1")
				mfcc_feat = mfcc(w.data,w.rate)
				d_mfcc_feat = delta(mfcc_feat, 2)
				dd_mfcc_feat = delta(d_mfcc_feat, 2)

				result=numpy.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
				result_1=numpy.concatenate((result,dd_mfcc_feat),axis=1)
				# print(result_1.shape)
				# temp.append(result_1[0:1119])
				temp.append(result_1)

	x_train = numpy.stack(temp)

	print(x_train)
	print(x_train.shape)
	print(type(x_train))

	x_train = x_train.reshape(-1, 39)

	print(x_train)
	print(x_train.shape)
	print(type(x_train))

	y_train=numpy.zeros(599500, dtype=numpy.int)

	index=0
	for i in range(0,10):
		y_train[index:(index+59950)]=i # 59950=1199*50
		index+=59950

	# y_train[0:59950]=0
	# y_train[59950:119900]=1
	# y_train[119900:179850]=2
	# y_train[179850:239800]=3
	# y_train[239800:299750]=4
	# y_train[299750:359700]=5
	# y_train[359700:419650]=6
	# y_train[419650:479600]=7
	# y_train[479600:539550]=8
	# y_train[539550:599500]=9

	return x_train, y_train

def getTestingData():
	temp=[]

	for k in range(1,11):
		for i in range(0,10):
				filename="./wav/"+str(k)+"/"+str(i)+"_5.wav"
				# print(filename)
				w= wavio.read(filename)
				# print("2")
				mfcc_feat = mfcc(w.data,w.rate)
				d_mfcc_feat = delta(mfcc_feat, 2)
				dd_mfcc_feat = delta(d_mfcc_feat, 2)

				result=numpy.concatenate((mfcc_feat,d_mfcc_feat ),axis=1)
				result_1=numpy.concatenate((result,dd_mfcc_feat),axis=1)
				# print(result_1.shape)
				# temp.append(result_1[0:1199])
				temp.append(result_1)

	x_test = numpy.stack(temp)

	print(x_test)
	print(x_test.shape)
	print(type(x_test))

	x_test = x_test.reshape(-1, 39)

	print(x_test)
	print(x_test.shape)
	print(type(x_test))

	y_test=numpy.zeros(119900, dtype=numpy.int)

	index=0
	for i in range(0,10):
		y_test[index:(index+11990)]=i # 11990=1199*10
		index+=11990

	# y_test[0:11990]=0
	# y_test[11990:23980]=1
	# y_test[23980:35970]=2
	# y_test[35970:47960]=3
	# y_test[47960:55950]=4
	# y_test[55950:71940]=5
	# y_test[71940:83930]=6
	# y_test[83930:95920]=7
	# y_test[95920:107910]=8
	# y_test[107910:119900]=9

	return x_test, y_test


# Checking if a File Exists
if os.path.isfile('./x_train.npy') and os.path.isfile('./y_train.npy'):
	x_train = numpy.load('x_train.npy')
	y_train = numpy.load('y_train.npy')
	#print(numpy.load('x_train.npy'))
	#print(numpy.loadtxt('x_train.txt'))
	#print(numpy.load('y_train.npy'))
	#print(numpy.loadtxt('y_train.txt'))
	print(x_train)
	print(x_train.shape)
	print(type(x_train))
else:
	x_train, y_train=getTrainingData()
	# #Binary data
	numpy.save('x_train.npy', x_train)
	# #Human readable data
	# numpy.savetxt('x_train.txt', x_train)

	# #Binary data
	numpy.save('y_train.npy', y_train)
	# #Human readable data
	# numpy.savetxt('y_train.txt', y_train)


# Checking if a File Exists
if os.path.isfile('./x_test.npy') and os.path.isfile('./y_test.npy'):
	x_test = numpy.load('x_test.npy')
	y_test = numpy.load('y_test.npy')
	#print(numpy.load('x_test.npy'))
	#print(numpy.loadtxt('x_test.txt'))
	#print(numpy.load('y_test.npy'))
	#print(numpy.loadtxt('y_test.txt'))
	print(x_test)
	print(x_test.shape)
	print(type(x_test))
else:
	x_test, y_test=getTestingData()
	# #Binary data
	numpy.save('x_test.npy', x_test)
	# #Human readable data
	# numpy.savetxt('x_test.txt', x_test)

	# #Binary data
	numpy.save('y_test.npy', y_test)
	# #Human readable data
	# numpy.savetxt('y_test.txt', y_test)

def pca(X, n_components):
    pca1 = PCA(n_components = n_components)
    pca1.fit(X)
    print(pca1.explained_variance_ratio_)
    print(pca1.explained_variance_)
    return pca1.transform(X)

def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40)
    model.fit(X)
    return model.transform(X)

def plot_scatter(x, labels, title, txt = False):
    plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    # txts = []
    if txt:
        for i in range(10):
            xtext, ytext = numpy.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            # txts.append(txt)
    plt.show()

layer_output_pca = pca(x_train, 2)
plot_scatter(layer_output_pca, y_train, "with pca", txt = True)

layer_output_tsne = tsne(layer_output_pca, 2)
plot_scatter(layer_output_tsne, y_train, "with tsne", txt = True)

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# model_adam = load_model('my_model.h5')
model_adam = Sequential()
model_adam.add(Dense(128, input_dim=39))
model_adam.add(Activation('relu'))
model_adam.add(Dense(256))
model_adam.add(Activation('relu'))
model_adam.add(Dense(512))
model_adam.add(Activation('relu'))
model_adam.add(Dense(10))
model_adam.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
# sgd = SGD(lr=0.01,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
model_adam.compile(loss='categorical_crossentropy',
				optimizer='Adam',
				metrics=['accuracy'])


'''Fit models and use validation_split=0.1 '''
history_adam = model_adam.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							shuffle=True,
                    		validation_split=0.1,
                    		callbacks=[early_stopping])

score = model_adam.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


'''Access the loss and accuracy in every epoch'''
loss_adam	= history_adam.history.get('loss')
acc_adam 	= history_adam.history.get('acc')

''' Access the performance on validation data '''
val_loss_adam = history_adam.history.get('val_loss')
val_acc_adam = history_adam.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(1)
plt.subplot(121)
plt.plot(range(len(loss_adam)), loss_adam,label='Training')
plt.plot(range(len(val_loss_adam)), val_loss_adam,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_adam)), acc_adam,label='Training')
plt.plot(range(len(val_acc_adam)), val_acc_adam,label='Validation')
plt.title('Accuracy')
plt.show()

model_adam.summary()

# print(x_train[119898].shape) #(39,)
# print(x_train[119898:119899].shape)# (1, 39)

x_predict = x_train[119899:119902]
predict_p = model_adam.predict(x_predict) #辨識為各類的機率
predict_c = predict_p.argmax(axis=1) #辨識為一類
print(predict_p)
print(predict_c)


# Remarks
# os.system("pause")

# pip install h5py
# import h5py
if os.path.isfile('./my_model.h5')==False:
	model_adam.save('my_model.h5')
# del model_adam

# plot_model(model_adam, show_shapes=True, to_file='model.png')