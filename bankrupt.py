import tensorflow as tf
import numpy as np
import threading

#TODO testar com o 5 datasets
# Leitura do CSV
dataset_path = "./iart_dataset/1year.tfrecords"
'''
dataset_path = './iart_dataset/1year.csv'
filename_queue = tf.train.string_input_producer([dataset_path], num_epochs=1, shuffle=False)
file_reader = tf.TextLineReader(skip_header_lines=1) #File CSV tem 1 linha de titulos, e necessario ignora-la
key, value = file_reader.read(filename_queue)
'''
'''
dataset_path    = "./iart_dataset/"
firstyear_file  = "1year.csv"

def read_label_file(file):
    f = open(file, "r")
    filepaths = []
    labels = []
    for line in f:
        elements = line.split(",")
        attributes = elements[:len(elements)-1]
        label = elements[len(elements)]
'''
def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader();
    key, value = reader.read(filename_queue);
    example, label = tf.parse_single_example(value)
    return example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=False)
    example, label = read_tfrecord(filename_queue)
    example_batch, label_batch = tf.train.batch([example, label], batch_size)
    return example_batch, label_batch


# Parametros
learning_rate = 0.01
n_hidden = 200
n_input = 64
n_output = 1
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

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
with tf.Session() as session:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    thread_coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=thread_coordinator)

    while True:
        try:
            #TODO Resolver isto
            '''
            batch_x = tf.train.batch([features], 1)
            batch_y = tf.train.batch([x65], 1)
            '''
            batch_x, batch_y = input_pipeline(dataset_path, 10, 1)
            _, c = session.run([optimizer, cost], feed_dict={inputs: batch_x, labels: batch_y})
            print(c)
            # Obter uma amostra
            '''
            exemplo, label = session.run([features, x65])

            print(exemplo)
            print(label)
            '''
        except tf.errors.OutOfRangeError:
            break

    thread_coordinator.request_stop()
    thread_coordinator.join(threads)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
