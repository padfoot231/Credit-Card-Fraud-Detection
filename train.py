import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
#from sklearn.linear_model import SGDClassifier
#import cPickle as pk
import tensorflow as tf

df = pd.read_csv('creditcard.csv')

X = df.iloc[:,:30].values
y = df.iloc[:,30].values
y_list = y.tolist()
for i in range(len(y_list)):
    y_list[i] = int(y_list[i])
n_values = np.max(y_list) + 1
y = np.eye(n_values)[y_list]
y = y.astype(float)
X = X.astype(float)
print(X.shape,y.shape)

X_train = X[:274807]
y_train = y[:274807]
X_test = X[274807:]
y_test = y[274807:]
batch = 50
num = y.shape[0]
num_train = num - 10000
# X=df.iloc[:,[1,2,3,4,5,6,7,9,10]].values
# y=df.iloc[:,8].values
# ID=df.iloc[:,0].values
# y_list = y.tolist()
# for i in range(len(y_list)):
#    y_list[i] = int(y_list[i])
# n_values = np.max(y_list) + 1
# y = np.eye(n_values)[y_list]
# y = y.astype(float)
# X = X.astype(float)
# training_set = np.concatenate((X,y),axis=1)
#np.random.shuffle(training_set)
print(X.shape,y.shape)
def weights(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def mean(l):
	return sum(l)/len(l)
x = tf.placeholder(tf.float32,[None,30])
y_ = tf.placeholder(tf.float32,[None,2])

W1 = weights([30,5])
b1 = bias([5])
hl1 = tf.nn.relu(tf.matmul(x,W1) + b1)
W2 = weights([5,2])
b2 = bias([2])
y_nn = tf.matmul(hl1,W2) + b2
y_soft = tf.nn.softmax(tf.matmul(hl1,W2) + b2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_nn,1) , tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for j in range(15):
		print('epoch',j)
		l = []
		for i in range(5496):
			#print('step:',i)
			x_batch = X[0:num_train][i*50:i*50+50,:]
			y_batch = y[0:num_train][i*50:i*50+50,:]
			sess.run(train_step,{x:x_batch,y_:y_batch})
			print('loss ',sess.run(cross_entropy,{x:x_batch,y_:y_batch}))
			# print('accuracy ',sess.run(accuracy,{x:x_batch,y_:y_batch}))
		for k in range(200):
			x_batch_test = X[num_train:][k*50:k*50+50,:]
			y_batch_test = y[num_train:][k*50:k*50+50,:]
			l.append(sess.run(accuracy,{x:x_batch_test,y_:y_batch_test}))
		print('testing accuracy',mean(l))			

# for k in xrange(bat_test):	
# 	X_batch_test=X_test[k*bst:k*bst+bst,:]
# 	y_batch_test=y_test[k*bst:k*bst+bst]
# 	sc=model.score(X_batch_test, y_batch_test)
# 	mean.append(sc)
