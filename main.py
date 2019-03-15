#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# Define global parameters
L2REG = 1e-3
EPOCHS = 30
BATCH_SIZE = 2
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4

NUM_CLASSES = 2 # road and no road classification
IMAGE_SHAPE = (160, 576)  # KITTI dataset uses 160x576 images
data_dir = '/data'
runs_dir = './runs'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights from vgg file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # grab graph and each layer by name
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return w1, keep, layer3, layer4, layer7

print("Test loading VGG: ")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Decoder Part

    ## 1 by 1 convolution
    layer7_conv_1x1_ = tf.layers.conv2d(vgg_layer7_out, num_classes,  kernel_size=1, strides=(1, 1), padding = 'same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(L2REG))


    # deconvolution
    # up sample by 2
    layer7_out_ = tf.layers.conv2d_transpose(layer7_conv_1x1_, num_classes, kernel_size=4, strides=(2, 2), padding = 'same',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer = tf.contrib.layers.l2_regularizer(L2REG))

    # Check lecture notes for size each layer 

    ## 1 by 1 of vgg layer 4
    layer4_conv_1x1_ = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1, 1), padding='same',
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(L2REG))

    # skip connection 
    layer4_out_ = tf.add(layer7_out_, layer4_conv_1x1_)

    
    layer3_in1_ =  tf.layers.conv2d_transpose(layer4_out_, num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                              kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                              kernel_regularizer = tf.contrib.layers.l2_regularizer(L2REG))

    layer3_in2_ = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1, 1), padding='same',
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer = tf.contrib.layers.l2_regularizer(L2REG))

    layer3_out_ = tf.add(layer3_in1_, layer3_in2_)


    # # upsample
    output_upsamp = tf.layers.conv2d_transpose(layer3_out_, num_classes, kernel_size=16, strides= (8, 8), padding= 'same',
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(L2REG))
    

    # # for debugging pring the dimension. index 1:3 for x and y dimension
    tf.Print(output_upsamp, [tf.shape(output_upsamp)])

    return output_upsamp
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Cross entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # robust optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)

    return logits, train, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    sess.run(tf.global_variables_initializer())
    # Implement function
    for epoch in range(epochs):
        print("Training EPOCH:", epoch)
        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed_dict)
            print("... Loss = ", loss)

tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        print("Loading VGG model ...")
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output =  layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)

        # Placeholders for tf
        label = tf.placeholder(tf.int32, shape=[None, None, None, NUM_CLASSES])
        learning_rate = tf.placeholder(tf.float32)

        logits, train_op, cross_entropy_loss = optimize(layer_output, label, learning_rate, NUM_CLASSES)

        # Train NN using the train_nn function
        print("Training the network ...")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, label, keep_prob, learning_rate)

        # Safe the trained model
        print("Saving the trained model...")
        saver = tf.train.Saver()
        saver.save(sess, './runs/TrainedModel.ckpt')

        # Save inference data using helper.save_inference_samples
        print("Save inference samples...")
        helper.save_inference_samples(runs_dir, data_dir, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
