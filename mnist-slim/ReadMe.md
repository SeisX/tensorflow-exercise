# Some tips
- tf.nn.softmax_cross_entropy_with_logits
    > The usage of this function should declare the arguments name. Refer to [Github issue](http://https://github.com/tflearn/tflearn/issues/639).

- ResourceExhaustedError: OOM when allocating tensor with shape[10000,20,28,28]
    > This is due to the memory limitaion, the proper advice is feed the network in small batches to the eval graph using *feed_dict* like the example does with the training data. Refer to [Github issue](https://github.com/tensorflow/tensorflow/issues/136).