import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../../data/mnist", one_hot=True)

print("Finished extracting data")
