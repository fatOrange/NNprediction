
"""
Computational Biology
protein secondary structure prediction
"""
import tensorflow as tf
import numpy as np

windowLen = 13
proTypes = 8 # number of protein type len(uniquepro)
strucTypes = 5 # number of structure type
inputLength = windowLen * proTypes #273
hiddenNeuron = 2000
trainfilename = 'RNA_structure.train'
testfilename = 'RNA_structure.test'
structype = ['.', '[', ']','(',')']
epoch = 1 # 周期

def getpro(filename): # get protein sequences from file
	proSeq = []
	strucSeq = []

	with open(filename, "r") as proFile: # open file
		protein = ""
		structure = ""
		for line in proFile:
			if line[0] != '<':
				protein += line[0]
				structure += line[2]
			elif line[1] == "e": # end of sequence
				proSeq.append(protein)
				strucSeq.append(structure)
				protein = ""
				structure = ""
	return proSeq, strucSeq

def countpro(X): # count number of unique protein （how many different character in seq）
	uniquepro = []
	uniquepro.append('-') # blank in sequence
	for i in range(len(X)):
		for char in X[i]:
			if char not in uniquepro:
				uniquepro.append(char)
	return uniquepro

def getXY(test, proteinSeq, uniquepro):# what this mean
	a = 0.0
	predictstruc = []
	newstrlen = len(proteinSeq) + (windowLen//2) * 2 # windowLen = 13 窗口
	newstr = ['-'] * newstrlen
	newstr[windowLen//2:len(proteinSeq)+windowLen//2] = proteinSeq #len(proteinSeq) = 91
	for i in range(len(proteinSeq)):
		trainX = newstr[i:windowLen+i]
		x = protonum(trainX, uniquepro)
		if test == 0:
			sess.run(train_op, feed_dict={X:x, Y:y})
		elif test == 1:
			# print trainX
			prediction = structype[sess.run(predict, feed_dict={X:x})[0]]
			# print('Prediction: ' + str(prediction))
			# print 'Actual: ' + str(structureSeq[i])
			predictstruc.append(prediction)
	return predictstruc

def protonum(trainX, uniquepro):
	num = [None]* windowLen
	stype = [None]* windowLen
	for i in range(len(uniquepro)):
		for j in range(windowLen):
			if trainX[j] == uniquepro[i]:
				num[j] = i
	num = np.array(num)
	num = num.reshape([1,windowLen])
	return num

"""feed forward net"""
# one-hot input for each amino acid
X = tf.placeholder(tf.int32, [None, windowLen]) #13
onehotIn = tf.one_hot(X, proTypes)#one_hot encode
onehotR = tf.reshape(onehotIn, [-1,inputLength])# -1 represent no matter how many rows,just "inputLength" col  inputLength = 273 1rows 273cols
# first hidden layer weights
w1 = tf.get_variable("w1", shape=[inputLength,hiddenNeuron], initializer=tf.contrib.layers.xavier_initializer()) # inputLength = 273;hiddenNeuron = 20000
# matrix multiplication to get hidden layer 1
h1 = tf.matmul(onehotR, w1)
h1R = tf.nn.relu(h1)

# output layer
w2 = tf.get_variable("w2", shape=[hiddenNeuron,strucTypes], initializer=tf.contrib.layers.xavier_initializer()) # 20000*3
# matrix multiplication to get hidden layer 2
h2 = tf.matmul(h1R, w2)
# Yhat = tf.matmul(h1R, w2)
Yhat = tf.nn.relu(h2)
# predict by choosing highest type
predict = tf.argmax(Yhat,1)

Y = tf.placeholder(tf.int32, [None, 1]) # supervised output
onehotOut = tf.one_hot(Y, strucTypes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Yhat, labels=onehotOut)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
correct_prediction = tf.equal(tf.argmax(onehotOut, 1), tf.argmax(Yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
	tf.global_variables_initializer().run()

	# train
	proteinLists, structureLists = getpro(trainfilename) # load protein sequences from file
	uniquepro = countpro(proteinLists)
	test = 0
	save_path = saver.restore(sess, "./model/my_test_model2")
	print("Model saved in path: %s" % save_path)
	# test
	a = 0.0  # for calculating accuracy
	t = 0.0
	test = 1
	print('Testing...')
	eproteinLists, estructureLists = getpro(testfilename) # load protein sequences from file

	RNA_seq = input("Please input a RNA sequence: ")
	predictstruc = getXY(test, RNA_seq, uniquepro)
	# for j in range(len(eproteinLists)):
	# 	print('Sequence ' + str(j+1))
	# 	a, t, predictstruc = getXY(test, eproteinLists[j], estructureLists[j], uniquepro)
	# 	a += a
	# 	t += t
	# 	print('Protein:    ' + RNA_seq)
	# 	print('Actual:     ' + str(estructureLists[j]))
	print('Prediction: ' + str(''.join(predictstruc)))
	# print('Accuracy: ' + str(a/t*100) + '%')
