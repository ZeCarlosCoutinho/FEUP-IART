import tensorflow as tf
import numpy as np

#TODO testar com o 5 datasets
dataset_path = './iart_dataset/1year.csv'

filename_queue = tf.train.string_input_producer([dataset_path])

file_reader = tf.TextLineReader(skip_header_lines=1) #File CSV tem 1 linha de titulos, e necessario ignora-la
key, value = file_reader.read(filename_queue)

column_default_values = [
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
[1.],[1.],[1.],[1.],[],
] #Default values dos atributos. TODO Arranjar um melhor valor para o default

#Input tensors
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65 = tf.decode_csv(value, record_defaults=column_default_values)

features = tf.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,
x21, x22, x23, x24, x25, x26, x27, x28, x29, x30,
x31, x32, x33, x34, x35, x36, x37, x38, x39, x40,
x41, x42, x43, x44, x45, x46, x47, x48, x49, x50,
x51, x52, x53, x54, x55, x56, x57, x58, x59, x60,
x61, x62, x63, x64])

with tf.Session() as session:
    thread_coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=thread_coordinator)

    for i in range(7000):
        # Obter uma amostra
        exemplo, label = session.run([features, x65])
        print(exemplo)
        print(label)

    thread_coordinator.request_stop()
    thread_coordinator.join(threads)
