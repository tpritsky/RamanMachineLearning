# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:57:45 2018

@author: TPritsky
"""

'''NOTE: Tomorrow try running this on different lab data. Find a way to only extract tissue types in training data set.'''

'''ML_Raman: This function performs machine learning on Raman data. Calls on functions in 
label_raman_for_ML file to produce labeled data for machine learning. Needs input of a Raman 
data file and a text file with tissue region labels. The raman data file is a mat file and 
needs to be properly formatted by extractVariables.m'''

'''# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' '''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np
import tensorflow as tf
sess = tf.Session()
from scipy.io import loadmat
#export CUDA_VISIBLE_DEVICES=''

from label_raman_for_ML import getTissueBoundaries, RamanInBounds, labelRamanData

#tf.device('/device:GPU:2')

#import label data file
#training_label_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180424\20180424-RAM-001.txt'
training_label_file = sys.argv[1]

#import raman data file
#training_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180424\20180424-RAM-001_VARIABLES_1' 
training_file = sys.argv[2]

#import test label file:
#test_label_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180330\20180330-RAM-001.txt'
test_label_file = sys.argv[3]

#import raman test data file
#test_file = r'C:\Users\TPritsky\Desktop\Intuitive_Project\Raman\20180330\20180330-RAM-001_VARIABLES'
test_file = sys.argv[4]

'''LoadMatFile: Loads matlab file and extracts variables. Make sure that the matlab file is in a format
                that is readable by using reformatRamanMatfileForPython.mat to adjust the matfile format. Returns a tuple
                of the form [output_data x_coordinates y_coordinates image]'''
def loadMatfile(inputFile):
    data = loadmat(inputFile) #load raman data file (make sure the variables are readable by python)

    #extract the variables from data
    output_data = data['data']    #get raman data array
    x = data['data_x']  #get x_coordinates of raman data
    y = data['data_y']  #get y_coordinates of raman data
    image = data['im']  #store the image data
    
    #return variables
    return (output_data, x, y, image)

#printDimensions: Prints dimensions of 2D numpy array
def printDimensions(data):
    print("Num Rows")
    print(len(data))
    print("Num Columns")
    print(len(data[0]))

'''labels_as_integers: converts an array of tuples (corresponding to tissue labels) into an
array of integers. Returns the integer array as labels_as_integers, as well as a dictionary called 
unique_labels that stores tuples (corresponding to tissue labels) as keys and their corresponding
integers as values. This dictionary is used by test_labels_as_integers to create an array of integers
corresponding to tissue labels in test data.'''
def labels_as_integers(labels):
    unique_labels = {}  #store unique labels and their integer equivalents
    labels_as_integers = [] #store each label as an integer
    #store a unique integer for each unique label. key = label; value = integer
    for i in range(len(labels)):
        integer_rep = 0
        if labels[i] not in unique_labels:
            unique_labels[labels[i]] = integer_rep
            integer_rep +=1
    #store integer equivalents of each label
    for i in labels:
        labels_as_integers.append(unique_labels[i])
    return labels_as_integers, unique_labels

'''test_labels_as_integers: converts an array of tuples (corresponding to tissue labels) into an
array of integers corresponding to tissue labels in test data. Returns the integer array as 
labels_as_integers.'''
def test_labels_as_integers(labels, key):
    unique_labels = key  #store unique labels and their integer equivalents
    labels_as_integers = [] #store each label as an integer
    
    '''store integer equivalent of each label if it is present in the training data.
    Uses key from the training data'''
    for i in labels:
        if(i in unique_labels):
            labels_as_integers.append(unique_labels[i])
    return labels_as_integers

def main():
    #load Raman data from mat file. Note x and y are 2D vectors with one element outer vector
    data, x, y, im = loadMatfile(training_file)
    
    #load test Raman data from mat file. Note x and y are 2D vectors with single element outer vector
    data_test, x_test, y_test, im_test = loadMatfile(test_file)#remove empty layers
    
    #convert to numpy array
    data = np.array(data)
    data_test = np.array(data_test)
    
    x = np.array(x)
    y = np.array(y)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    #remove empty layers
    x = np.squeeze(x)
    y = np.squeeze(y)
    x_test = np.squeeze(x_test)
    y_test = np.squeeze(y_test)
    
    '''transpose data. Data must be transposed so that dimensions
    are samples (rows) by features (columns). This is verified by
    printDimensions.'''
    data = data.transpose()
    data_test = data_test.transpose()
            
    #test
    print("data shape")
    print(int(data.shape[0]))
    
    #TEST
    print("data length")
    print(len(data))
    print("test data len")
    print(len(data_test))
    
    #get number of Tissues from user
    #number_tissues = input("Input the number of tissues: ")
    
    '''Set training/test data labels: getTissueBoundaries returns a dictionary with key
    equal to tissue type and a value equal to a list of polygons bounding that tissue.'''
    polygon_dict = getTissueBoundaries(training_label_file)
    polygon_dict_test = getTissueBoundaries(test_label_file)
    
    '''List all labeled data: returns three vectors - labeled_x/labeled_y, which contain x/y 
    coordinates of labeled data and labels, which contains the corresponding tissue labels'''
    labeled_x, labeled_y, labels = RamanInBounds(x, y, polygon_dict)
    labeled_x_test, labeled_y_test, labels_test = RamanInBounds(x_test, y_test, polygon_dict_test)

    '''return a 2D numpy array (labeled_raman_data), which contains all labeled raman data. This list
    contains only raman data, and the corresponding labels are found in labels (a 1D list)'''
    labeled_data = labelRamanData(data, x, labeled_x, y, labeled_y)
    labeled_data_test = labelRamanData(data_test, x_test, labeled_x_test, y_test, labeled_y_test)
    
    #TEST
    print("labels")
    print(labels)
    print("Original Dimensions")
    printDimensions (labeled_data)
    print("New Dimensions")
    printDimensions (labeled_data_test)
    
    #get number of labels for Machine Learning
    num_tissues = len(polygon_dict)
    
    #verification that training and test data have same number of tissues. No Longer Needed
    '''if(len(polygon_dict) != len(polygon_dict_test)):
        print("Tissue types are not synonymous between training and test data. Please re-run, since datasets are produced randomly.")
        quit()'''
    
    #remove labeled data not in training set from test data set
    
    '''convert training label vector to a vector of integers for softmax
    through labels_as_integers method. Get dictionary of integer labels
    corresponding to a given tissue type'''
    integer_labels, label_to_integer_key = labels_as_integers(labels)
    
    '''Convert test label vector to a vector of integers for softmax through test_labels_as_integers. 
    Use label_to_integer_key created by labels_as_integers function to get correct integers 
    corresponding to labels'''
    integer_labels_test = test_labels_as_integers(labels_test, label_to_integer_key)
    
    #convert the array of int labels to numpy array
    labels_np = np.array(integer_labels).astype(dtype=np.uint8)
    labels_np_test = np.array(integer_labels_test).astype(dtype=np.uint8)
    
    #convert int numpy array into onehot matrix
    labels_onehot = (np.arange(num_tissues) == labels_np[:,None]).astype(np.float32)
    labels_onehot_test = (np.arange(num_tissues) == labels_np_test[:,None]).astype(np.float32)
    
    #TEST
    print("Label name")
    print(labels_np[7])
    
    '''Reshape labels_onehot array to remove extra dimension. This produces the final list of
    labels that will be used for machine learning'''
    labels_onehot = np.squeeze(labels_onehot)
    labels_onehot_test = np.squeeze(labels_onehot_test)
    
    #TEST: print original and test data dimensions
    print("Original Data Dimensions")
    print(labeled_data.shape)
    print("Original Label Dimensions")
    print(labels_onehot.shape)
    print("New Data Dimensions")
    print(labeled_data_test.shape)
    print("New Label Dimensions")
    print(labels_onehot_test.shape)
    
    with(tf.device('/device:GPU:0')):
        '''Create placeholder to store pixel data. The placeholder
        can store a variable number of samples (indicated by None)
        each with 256 features.'''
        x = tf.placeholder(tf.float32, [None, 256])
        
        '''Create placeholder to store pixel labels. The placeholder
        can store a variable number of samples (indicated by None)
        each with 'num_tissues' possible tissue labels.'''
        y_ = tf.placeholder(tf.float32, [None, num_tissues])
        
        #initialize weight and bias values to zero. Num_tissues stores the total number of tissues.
        W = tf.Variable(tf.zeros([256,num_tissues]))
        b = tf.Variable(tf.zeros([num_tissues]))
        
        #initialize variables simultaneously
        sess.run(tf.global_variables_initializer())
    
        #implement regression model: Multiply image input data 'x' by weigths 'w' and add bias 'b'
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        
        '''specify loss function: Loss indicates how bad the model's prediction was on a single example; 
        we try to minimize that while training across all the examples.'''
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        
        '''Train the Model'''
        
        #used gradient descent optimization algorithm to minimize loss
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
    #shape labels
    #labels = np.array(labels)
    
    #print("Data Types")
    for i in labels:
        if(type(i) != str):
            print(type(i))
    
    #run the train_step function 1000 times to train the model
    '''MAY WANT TO RANDOMIZE BATCH ELEMENTS'''
    
    BATCH_SIZE = 128
    
    #tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
    #tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
    
    for i in range(1000):
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_X = labeled_data[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
        batch_Y = labels_onehot[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
        sess.run(train_step, feed_dict={x: batch_X, y_: batch_Y})
        
        '''Not sure how to include batch/not sure how this training works?: 
        batch = labeled_raman_data.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        '''
        
    '''evaluate the model with test data: data-> labeled_data_test; labels-> labels_onehot_test'''
    
    #check if predicted value matches truth. Return list of booleans
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #record average accuracy as percentage of correct predicitions over total predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #test run to determine model accuracy given the test data
    print(sess.run(accuracy, feed_dict={x: labeled_data_test, y_ : labels_onehot_test}))

main()
    
