import os.path
import sys
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


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return vgg_input, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # conv 1x1
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, padding = 'same',
                   kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # up sampling
    up_sampling_1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding = 'same',
                        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # skip layer, first make it depth 2, then scaling before adding
    layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding = 'same',
             kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    layer4 = tf.multiply(layer4, 0.01)
    skip_1 = tf.add(layer4, up_sampling_1)

    # up sampling
    up_sampling_2 = tf.layers.conv2d_transpose(skip_1, num_classes, 4, 2, padding = 'same',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # skip layer, first make it depth 2, then scaling before adding
    layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding = 'same',
             kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    layer3 = tf.multiply(layer3, 0.0001)
    skip_2 = tf.add(layer3, up_sampling_2)

    # up sampling
    up_sampling_3 = tf.layers.conv2d_transpose(skip_2, num_classes, 16, 8, padding = 'same',
                    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    return up_sampling_3
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # reshape logits and label to 2D
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = 'logits')
    label = tf.reshape(correct_label, (-1, num_classes))

    # calculate accuracy from logits and label
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                        logits = logits, labels = label))
    # add l2 loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + sum(reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

    train_op = optimizer.minimize(loss)

    return logits, train_op, loss, accuracy
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
             cross_entropy_loss, input_image, accuracy,
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
    # TODO: Implement function
    print("Training...")
    for epoch in range(epochs):
        print("==============epoch: {}==============".format(epoch))

        acc_total = 0
        img_cnt = 0
        for image, label in get_batches_fn(batch_size):
            _, loss, acc = sess.run([train_op, cross_entropy_loss, accuracy], 
                               feed_dict= {input_image: image, correct_label:label,
                               keep_prob: 1.0, learning_rate: 0.0005})

            print("loss: {}".format(loss))

            acc_total += acc * len(image)
            img_cnt += len(image)

        print("accuracy in this epoch: {:.3f}".format(acc_total / img_cnt))
#tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.int32, (None, image_shape[0], image_shape[1], num_classes))
    epochs = 20
    batch_size = 4
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        layer_output = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss, accuracy = optimize(layer_output, labels,
                                                   learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, 
                 cross_entropy_loss, input_image, accuracy,
                 labels, keep_prob, learning_rate)

        # save model for later inference
        saver.save(sess, "./ss")

def infer():
    '''
    function to do inference using a saved model
    '''
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./ss.meta')
        saver.restore(sess, './ss')
        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0');
        logits = graph.get_tensor_by_name('logits:0')

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                                      keep_prob, input_image)

def video():
    '''
    function to process a video
    '''
    video_file = "./data/project_video.mp4"
    save_file = "./data/processed_video.mp4"
    image_shape = (160, 576)
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('./ss.meta')
        saver.restore(sess, './ss')
        graph = tf.get_default_graph()
        input_image = graph.get_tensor_by_name('image_input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0');
        logits = graph.get_tensor_by_name('logits:0')

        helper.process_video_file(video_file, save_file, sess, image_shape, logits,
                                 keep_prob, input_image)


if __name__ == '__main__':
     if len(sys.argv) != 2:
         print("usage: python main.py train|infer|video")
     elif sys.argv[1] == "train":
         run()
     elif sys.argv[1] == "infer":
         infer()
     elif sys.argv[1] == "video":
         video()
     else:
         print("usage: python main.py train|infer|video")

