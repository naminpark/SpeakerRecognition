#__author__ = 'naminpark'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from mfcc_feature_extraction import *
import copy


TRpath = './VAD/TRAIN_dependant/'
TEpath = './VAD/TEST_dependant/'


DR=data_RW()

GMix=1
n_mfcc =20
n_class =15



DR.setVariable(n_mfcc,n_class,GMix)


Trfeat = DR.csvRead(TRpath+"input_feat_TRAIN.csv")
Trlabel= DR.csvRead(TRpath+"input_label_TRAIN.csv")

Tefeat = DR.csvRead(TEpath+"input_feat_TEST.csv")
Telabel= DR.csvRead(TEpath+"input_label_TEST.csv")


DR.setValue(Trfeat,Trlabel)

class Model:

    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

        # Xavier Init
    def xavier_init(self,n_inputs, n_outputs, uniform=True):
      """Set the parameter initialization using the method described.
      This method is designed to keep the scale of the gradients roughly the same
      in all layers.
      Xavier Glorot and Yoshua Bengio (2010):
               Understanding the difficulty of training deep feedforward neural
               networks. International conference on artificial intelligence and
               statistics.
      Args:
        n_inputs: The number of input nodes into each output.
        n_outputs: The number of output nodes for each input.
        uniform: If true use a uniform distribution, otherwise use a normal.
      Returns:
        An initializer.
      """
      if uniform:
           # 6 was used in the paper.
           init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
           return tf.random_uniform_initializer(-init_range, init_range)
      else:
           # 3 gives us approximately the same limits as above since this repicks
           # values greater than 2 standard deviations from the mean.
           stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
           return tf.truncated_normal_initializer(stddev=stddev)




    def _build_net(self):

        with tf.variable_scope(self.name):

            beta = 0.01
            # Network Parameters
            n_input    = n_mfcc*2*GMix # MFCC Feature
            n_hidden_1 = 2048 # 1st layer num features
            n_hidden_2 = 2048 # 2nd layer num features
            n_hidden_3 = 1024 # 3rd layer num features
            n_hidden_4 = 512 # 4th layer num features
            n_hidden_5 = 512 # 5th layer num features
            n_classes  = 15 # speaker classes



            self.training = tf.placeholder(tf.bool)

            # tf Graph input
            self.x = tf.placeholder("float", [None, n_input])
            self.y = tf.placeholder("float", [None, n_classes])

            self.dropout_keep_prob = tf.placeholder("float")

            #lr = tf.placeholder("float")

            scale1= tf.Variable(tf.ones([n_hidden_1]))
            beta1 = tf.Variable(tf.zeros([n_hidden_1]))

            scale2= tf.Variable(tf.ones([n_hidden_3]))
            beta2 = tf.Variable(tf.zeros([n_hidden_3]))

            scale3= tf.Variable(tf.ones([n_hidden_4]))
            beta3 = tf.Variable(tf.zeros([n_hidden_4]))


                        # Store layers weight & bias
            weights = {
                'h1': tf.get_variable("h1", shape=[n_input, n_hidden_1],    initializer=self.xavier_init(n_input,n_hidden_1)),
                'h2': tf.get_variable("h2", shape=[n_hidden_1, n_hidden_2], initializer=self.xavier_init(n_hidden_1,n_hidden_2)),
                'h3': tf.get_variable("h3", shape=[n_hidden_2, n_hidden_3], initializer=self.xavier_init(n_hidden_2,n_hidden_3)),
                'h4': tf.get_variable("h4", shape=[n_hidden_3, n_hidden_4], initializer=self.xavier_init(n_hidden_3,n_hidden_4)),
                'h5': tf.get_variable("h5", shape=[n_hidden_4, n_hidden_5], initializer=self.xavier_init(n_hidden_4,n_hidden_5)),

                'out': tf.get_variable("out", shape=[n_hidden_5, n_classes], initializer=self.xavier_init(n_hidden_5,n_classes))
            }
            biases = {
                'b1': tf.Variable(tf.zeros([n_hidden_1])),
                'b2': tf.Variable(tf.zeros([n_hidden_2])),
                'b3': tf.Variable(tf.zeros([n_hidden_3])),
                'b4': tf.Variable(tf.zeros([n_hidden_4])),
                'b5': tf.Variable(tf.zeros([n_hidden_5])),
                'out': tf.Variable(tf.zeros([n_classes]))
            }


            layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])

            _mean1, _var1 = tf.nn.moments(layer_1, [0])
            BN1 = tf.nn.batch_normalization(layer_1, _mean1, _var1, beta1,scale1, 0.0001)

            layer_1=tf.nn.relu(BN1)
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

            layer_3=tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])

            _mean2, _var2 = tf.nn.moments(layer_3, [0])
            BN2 = tf.nn.batch_normalization(layer_3, _mean2, _var2, beta2,scale2, 0.0001)

            layer_3=tf.nn.relu(BN2)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            _mean3, _var3 = tf.nn.moments(layer_4, [0])
            BN3 = tf.nn.batch_normalization(layer_4, _mean3, _var3, beta3,scale3, 0.0001)
            layer_4=tf.nn.relu(BN3)

            layer_4 = tf.nn.dropout(tf.nn.relu(layer_4), self.dropout_keep_prob)

            layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))

            self.pred= (tf.nn.softmax(tf.matmul(layer_5, weights['out']) + biases['out'])) # No need to use softmax??


        inf=1e-7
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred+inf))) +beta * tf.nn.l2_loss(weights['h1'])+beta * tf.nn.l2_loss(weights['h2'])
        +beta * tf.nn.l2_loss(weights['h3'])+beta * tf.nn.l2_loss(weights['h4'])+beta * tf.nn.l2_loss(weights['out'])
        +beta * tf.nn.l2_loss(biases['b1'])+beta * tf.nn.l2_loss(biases['b2'])+beta * tf.nn.l2_loss(biases['b3'])
        +beta * tf.nn.l2_loss(biases['b4'])+beta * tf.nn.l2_loss(biases['b5'])+beta * tf.nn.l2_loss(biases['out'])# Softmax loss

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost) # Adam Optimizer
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.8).minimize(cost) # Adam Optimizer

        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def predict(self, x_test, training=False):
        return self.sess.run(self.pred,feed_dict={self.x: x_test, self.training: training, self.dropout_keep_prob:1.0})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test,self.y: y_test, self.training: training,self.dropout_keep_prob:1.0})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data, self.y: y_data, self.training: training,self.dropout_keep_prob:0.5})


# Parameters

training_epochs = 30
batch_size      = 400
display_step    = 10


sess = tf.Session()

models =[]

num_models=3

for m in range(num_models):
    models.append(Model(sess,"model"+str(m)))

# Initializing the variables
sess.run(tf.global_variables_initializer())



# Training cycle
for epoch in range(training_epochs):

    avg_cost_list =np.zeros(len(models))
    total_batch = int((DR.dataX.shape[0])/batch_size)

    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = DR.next_batch(i,batch_size)
        # Fit training using batch data

        for m_idx, m in enumerate(models):
            c,_=m.train(batch_xs,batch_ys)
            avg_cost_list[m_idx] +=c/total_batch


    if epoch % display_step == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print ("Optimization Finished!")



test_size = len(Telabel)

predictions = np.zeros(test_size * n_class).reshape(test_size, n_class)
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        Tefeat, Telabel))
    p = m.predict(Tefeat)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(Telabel, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))



#test_acc = sess.run(accuracy, feed_dict={x: Tefeat, y: Telabel, dropout_keep_prob:1.})
#print ("Training accuracy: %.3f" % (test_acc))


#from sklearn.metrics import roc_curve, auc

#result= sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.})

#fpr, tpr, threshold = roc_curve(Telabel, sess.run(pred, feed_dict={x: Tefeat,  dropout_keep_prob:1.}), pos_label=1)
#EER = threshold(np.argmin(abs(tpr-fpr)))
