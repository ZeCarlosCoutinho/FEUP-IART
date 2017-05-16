import tensorflow as tf
import numpy as np
import threading
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

#TODO testar com o 5 datasets
# Leitura do CSV
#dataset_path = "./iart_dataset/1year.tfrecords"

# Ler o ficheiro CSV
dataset_path = './iart_dataset/1year.csv'
raw_dataset = pd.read_csv(dataset_path)

if raw_dataset is None:
    print("Error reading .csv file")
else:
    print("Raw data read")

'''
def read_csv(filename_queue, default_values):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, values_string = reader.read(filename_queue)
    all_values = tf.decode_csv(values_string, record_defaults=default_values)
    values = all_values[:64]
    label = all_values[64]
    return values, label

def input_pipeline(filenames, batch_size, default_values, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    values, label = read_csv(filename_queue, default_values)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    values_batch, label_batch = tf.train.shuffle_batch(
        [values, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue
    )
    return values_batch, label_batch
'''

# Parametros
CLASS_NAME      = "class"
ATTRIBUTES      = [i for i in raw_dataset.keys().tolist() if i != CLASS_NAME]
NUM_EXAMPLES    = raw_dataset.shape[0]
RANDOM_SEED     = 50
TEST_SIZE       = 0.1
TRAIN_SIZE      = int(NUM_EXAMPLES * (1 - TEST_SIZE))
EPOCHS          = 100
DISPLAY_STEP    = 10
n_input         = raw_dataset.shape[1] - 1
n_output        = raw_dataset['class'].unique().shape[0] #o unique devolve um array de valores diferentes, e o shape[0] e o tamanho desse array
n_hidden        = 200

learning_rate   = 0.5
batch_size      = 100
column_default_values = [
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[],
] #Default values dos atributos. TODO Arranjar um melhor valor para o default
# Attr21 - E um ratio a comprar sales atuais com as do ano passado. Default = 1

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

#inputs_batch, labels_batch = input_pipeline([dataset_path], batch_size, column_default_values)

# Construcao do modelo
prediction = multilayer_perceptron(inputs, weights, biases)

# Perda e optimizador da perda
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Precisao
correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

init = tf.initialize_all_variables()

print("Network built")

#Input tensors
'''
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65 = tf.decode_csv(value, record_defaults=column_default_values)

features = tf.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
x51, x52, x53, x54, x55, x56, x57, x58, x59, x60,
x61, x62, x63, x64])
'''
print("Training...")
with tf.Session() as session:
    session.run(init)
    '''
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    thread_coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=thread_coordinator)

    x_input, y_input = session.run([inputs_batch, labels_batch])

    session.run(accuracy, feed_dict={
        inputs: x_input,
        labels: y_input
    })
    '''
    for epoch in xrange(EPOCHS):
            #TODO Resolver isto
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
                print("Epoch: %03d/%03d cost: %.9f" % (epoch, EPOCHS, average_cost))
                training_accuracy = session.run(accuracy, feed_dict={inputs:batch_examples, labels:batch_labels})
                print("Training accuracy: %.3f" % (training_accuracy))


    print("Ended training.")
    print("Begin testing...")

    test_accuracy = session.run(accuracy, feed_dict={inputs: examples_test, labels: labels_test})
    print("Test accuracy: %.6f" % (test_accuracy))
    session.close()
    print("Session closed")
'''
            x_input, y_input = session.run([inputs_batch, labels_batch])
            y_input = np.reshape(y_input, (n_hidden, n_output))
            _, loss = session.run([optimizer, cost], feed_dict={
                inputs: x_input,
                labels: y_input
            })

            if i % 500 == 0:
                print('iter:%d - loss:%f' % (i, loss))
            # Obter uma amostra

            exemplo, label = session.run([features, x65])

            print(exemplo)
            print(label)

    x_input, y_input = session.run([inputs_batch, labels_batch])


    thread_coordinator.request_stop()
    thread_coordinator.join(threads)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
'''
#session.close()
