#!/usr/bin/env python
import numpy
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

x_train = numpy.load('x_train.npy')
y_train = numpy.load('y_train.npy')
#print(numpy.load('x_train.npy'))
#print(numpy.loadtxt('x_train.txt'))
#print(numpy.load('y_train.npy'))
#print(numpy.loadtxt('y_train.txt'))
# print(x_train)
# print(x_train.shape)
# print(type(x_train))

x_test = numpy.load('x_test.npy')
y_test = numpy.load('y_test.npy')
#print(numpy.load('x_test.npy'))
#print(numpy.loadtxt('x_test.txt'))
#print(numpy.load('y_test.npy'))
#print(numpy.loadtxt('y_test.txt'))
# print(x_test)
# print(x_test.shape)
# print(type(x_test))

model_ce = load_model('my_model.h5')

# bottleneck_layer=model_ce.get_layer(index=6)

# print(bottleneck_layer.output)

from keras import backend as K
# with a Sequential model
get_5th_layer_output = K.function([model_ce.layers[0].input],
                                  [model_ce.layers[5].output])


layer_output = get_5th_layer_output([x_train])[0]
# print(layer_output)
# print(layer_output.shape) # (599500, 512)

layer_output_1 = get_5th_layer_output([x_test])[0]
# print(layer_output)
# print(layer_output_1.shape) # (119900, 512)

def pca(X, n_components):
    pca1 = PCA(n_components = n_components)
    pca1.fit(X)
    print(pca1.explained_variance_ratio_)
    print(pca1.explained_variance_)
    return pca1.transform(X)

def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40, init='pca')
    return model.fit_transform(X)

def plot_scatter(x, labels, title, txt = False):
    plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    txts = []
    if txt:
        for i in range(10):
            xtext, ytext = numpy.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.show()

# layer_output_pca = pca(layer_output, 2)
# plot_scatter(layer_output_pca, y_train, "with pca", txt = True)

# layer_output_tsne = tsne(layer_output, 2)
# plot_scatter(layer_output_tsne, y_train, "with tsne", txt = True)

# layer_output_1_pca = pca(layer_output_1, 2)
# plot_scatter(layer_output_1_pca, y_test, "with pca", txt = True)

# layer_output_1_tsne = tsne(layer_output_1, 2)
# plot_scatter(layer_output_1_tsne, y_test, "with tsne", txt = True)

temp_singer=numpy.empty(shape=[0, 512])
temp_train=numpy.empty(shape=[0, 512])


index=0
for i in range(0,10):
	temp_singer=numpy.mean(layer_output[index:(index+59950)], axis=0)
	index+=59950
	# print(temp_singer)
	# print(temp_singer.shape)
	# print(type(temp_singer))
	temp_train=numpy.vstack((temp_train,temp_singer))


train_label=numpy.zeros(10, dtype=numpy.int)
for i in range(0,10):
	train_label[i]=i # 11990=1199*10

temp_train_pca = pca(temp_train, 2)
plot_scatter(temp_train_pca, train_label, "with pca", txt = True)

temp_train_tsne = tsne(temp_train, 2)
plot_scatter(temp_train_tsne, train_label, "with tsne", txt = True)

# print(temp_train)
# print(temp_train.shape)
# print(type(temp_train))

# print(temp_train[0])
# print(temp_train[0].shape)
# print(type(temp_train[0]))


temp_song=numpy.empty(shape=[0, 512])
temp_test=numpy.empty(shape=[0, 512])


index=0
for i in range(0,100):
	temp_song=numpy.mean(layer_output_1[index:(index+1199)], axis=0)
	index+=1199
	# print(temp_song)
	# print(temp_song.shape)
	# print(type(temp_song))
	temp_test=numpy.vstack((temp_test,temp_song))

test_label=numpy.zeros(100, dtype=numpy.int)
index=0
for i in range(0,10):
	test_label[index:(index+10)]=i # 11990=1199*10
	index+=10


temp_test_pca = pca(temp_test, 2)
plot_scatter(temp_test_pca, test_label, "with pca", txt = True)

temp_test_tsne = tsne(temp_test, 2)
plot_scatter(temp_test_tsne, test_label, "with tsne", txt = True)

# print(temp_test)
# print(temp_test.shape)
# print(type(temp_test))

# print(temp_test[0])
# print(temp_test[0].shape)
# print(type(temp_test[0]))

min_dist=numpy.empty(shape=[0])

for i in range(0,100):
	model_dist=numpy.empty(shape=[0])
	for j in range(0,10):
		dist = numpy.linalg.norm(temp_test[i]-temp_train[j])
		# print(dist)
		model_dist=numpy.hstack((model_dist,dist))

	# print(model_dist)
	# print(model_dist.shape)
	# print(type(model_dist))
	# print(numpy.argmin(model_dist))
	min_dist=numpy.hstack((min_dist,numpy.argmin(model_dist)))

print(min_dist)
print(min_dist.shape)
print(type(min_dist))

min_dist = min_dist.reshape(-1, 10)

print(min_dist)
print(min_dist.shape)
print(type(min_dist))

print(min_dist[0][0])

numpy.save('min_dist.npy', min_dist)