import tensorflow as tf
import numpy as np
import threading
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

#TODO testar com o 5 datasets

# Ler o ficheiro CSV
dataset_path = './iart_dataset/1year.csv'
dataset_path2 = './iart_dataset/2year.csv'
dataset_pathOversampled = './iart_dataset/allyearsUndersampled.csv'
raw_dataset = pd.read_csv(dataset_pathOversampled)

if raw_dataset is None:
    print("Error reading .csv file")
else:
    print("Raw data read")

# Parametros
CLASS_NAME      = "class"
ATTRIBUTES      = [i for i in raw_dataset.keys().tolist() if i != CLASS_NAME]
NUM_EXAMPLES    = raw_dataset.shape[0]
RANDOM_SEED     = 50
TEST_SIZE       = 0.2
TRAIN_SIZE      = int(NUM_EXAMPLES * (1 - TEST_SIZE))
EPOCHS          = 8000
DISPLAY_STEP    = 10
n_input         = raw_dataset.shape[1] - 1
n_output        = raw_dataset['class'].unique().shape[0] #o unique devolve um array de valores diferentes, e o shape[0] e o tamanho desse array
n_hidden        = 33

learning_rate   = 0.011
batch_size      = 100

# Carregar os dados
examples_raw = raw_dataset[ATTRIBUTES].get_values()
labels = raw_dataset[CLASS_NAME].get_values()

# Pre-processamento do dataset
#   One hot encoding
labels_onehot = np.zeros((NUM_EXAMPLES, n_output))
labels_onehot[np.arange(NUM_EXAMPLES), labels] = 1

#   Prencher os missing values
imputer = Imputer(missing_values = 'NaN', strategy="mean", axis=1)
examples_processed = imputer.fit_transform(examples_raw)
print("Data preprocessed")

# Dividir os exemplos em sets de treino e teste
examples_train, examples_test, labels_train, labels_test = train_test_split(examples_processed,
                                                                            labels_onehot,
                                                                            test_size = TEST_SIZE,
                                                                            random_state=RANDOM_SEED)

print("Data split into training and testing sets")


# Modelo
def combination_function(input_values, weights, biases):
    return tf.add(tf.matmul(input_values, weights), biases)

def activation_function(input_values):
    return tf.nn.sigmoid(input_values)

def multilayer_perceptron(input_layer, weights, biases):
    hidden_layer = activation_function(combination_function(input_layer, weights['hidden'], biases['hidden']))
    out_layer = activation_function(combination_function(hidden_layer, weights['out'], biases['out']))
    return out_layer


# Pesos e Biases
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), #Matriz com valores aleatorios segundo uma distribuicao normal
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Input do grafo
inputs = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_output])

# Construcao do modelo
prediction = multilayer_perceptron(inputs, weights, biases)

# Perda e optimizador da perda
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Precisao
correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

init = tf.initialize_all_variables()

print("Network built")
print("Training...")
with tf.Session() as session:
    session.run(init)
    training_accuracy = 0;
    epoch = 0;
    while training_accuracy < 0.95:
    #for epoch in xrange(EPOCHS):
            average_cost = 0;
            total_batch = int(examples_train.shape[0]/batch_size)

            for i in xrange(total_batch):
                #Get batches with random samples from training set
                random_indexes = np.random.randint(int(TRAIN_SIZE), size= batch_size)
                batch_examples = examples_train[random_indexes, :]
                batch_labels = labels_train[random_indexes, :]

                session.run(optimizer, feed_dict={inputs: batch_examples, labels: batch_labels})
                average_cost += session.run(cost, feed_dict={inputs: batch_examples, labels: batch_labels})/total_batch
            if epoch % DISPLAY_STEP == 0:
                #print("Epoch: %03d/%03d cost: %.9f" % (epoch, EPOCHS, average_cost))
                print("Epoch: %03d cost: %.9f" % (epoch, average_cost))
                training_accuracy = session.run(accuracy, feed_dict={inputs:batch_examples, labels:batch_labels})
                print("Training accuracy: %.3f" % (training_accuracy))
            epoch += 1

    print("Ended training.")
    print("Begin testing...")

    test_accuracy = session.run(accuracy, feed_dict={inputs: examples_test, labels: labels_test})
    print("Test accuracy: %.6f" % (test_accuracy))

    session.close()
    print("Session closed")
