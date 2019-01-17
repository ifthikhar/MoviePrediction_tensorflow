

import tensorflow as tf
import os
import sys


tf.logging.set_verbosity(tf.logging.INFO)

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
tf.logging.info("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

PATH = "/tmp/tf_custom_model12"

# Fetch and store Training and Test dataset files
FILE_TRAIN = "movie_training.csv"
FILE_TEST = "movie_test.csv"

# The CSV features in our training & test data
feature_names = ['num_critic_for_reviews',
'revenue',
'num_voted_users',
'cast_total_facebook_likes',
'num_user_for_reviews',
'budget',
'imdb_score',
'movie_facebook_likes',
'vote_average',
'vote_count',
'cast_power']

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
        .skip(1)  # Skip header row
        .map(decode_csv, num_parallel_calls=2)  # Decode each line
        .cache() # Warning: Caches entire dataset, can cause out of memory
        .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
        .repeat(repeat_count)    # Repeats dataset this # times
        .batch(32)
        .prefetch(1)  # Make sure you always have 1 batch ready to serve
    )
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode):    # And instance of tf.estimator.ModeKeys, see below

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    # All our inputs are feature columns of type numeric_column
    feature_columns = [
        tf.feature_column.numeric_column(feature_names[0]),
        tf.feature_column.numeric_column(feature_names[1]),
        tf.feature_column.numeric_column(feature_names[2]),
        tf.feature_column.numeric_column(feature_names[3]),
        tf.feature_column.numeric_column(feature_names[4]),
        tf.feature_column.numeric_column(feature_names[5]),
        tf.feature_column.numeric_column(feature_names[6]),
        tf.feature_column.numeric_column(feature_names[7]),
        tf.feature_column.numeric_column(feature_names[8]),
        tf.feature_column.numeric_column(feature_names[9]),
        tf.feature_column.numeric_column(feature_names[10])
    ]

    # Create the layer of input
    input_layer = tf.feature_column.input_layer(features, feature_columns)

    # Definition of hidden layer: h1
    # We implement it as a fully-connected layer (tf.layers.dense)
    # Has 10 neurons, and uses ReLU as the activation function
    # Takes input_layer as input
    h1 = tf.layers.Dense(1024, activation=tf.nn.relu)(input_layer)

    # Definition of hidden layer: h2 (this is the logits layer)
    # Similar to h1, but takes h1 as input
    h2 = tf.layers.Dense(512, activation=tf.nn.relu)(h1)
    h3 = tf.layers.Dense(256, activation=tf.nn.relu)(h2)
    h4 = tf.layers.Dense(128, activation=tf.nn.relu)(h3)



    # Output 'logits' layer is three number = probability distribution
    # between Iris Setosa, Versicolor, and Viginica
    logits = tf.layers.Dense(5)(h4)

    # class_ids will be the model prediction for the class (movie type)
    # The output node with the highest value is our prediction
    predictions = { 'class_ids': tf.argmax(input=logits, axis=1) }

    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation and Training mode

    # Calculate the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Calculate the accuracy between the true labels, and our predictions
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'my_accuracy': accuracy})

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode

    # Default optimizer for DNNClassifier: Adagrad with learning rate=0.05
    # Our objective (train_op) is to minimize loss
    # Provide global step counter (used to count gradient updates)
    optimizer = tf.train.AdagradOptimizer(0.05)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step())

    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops
    tf.summary.scalar('my_accuracy', accuracy[1])

    # Return training operations: loss and train_op
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)

# Create a custom estimator using my_model_fn to define the model
tf.logging.info("Before classifier construction")

# TODO monitor the contribution made by the tensors
# validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#     test_set.data,
#     test_set.target,
#     every_n_steps=50)

classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=PATH)  # Path to where checkpoints etc are stored
tf.logging.info("...done constructing classifier")

# 500 epochs = 500 * 120 records [60000] = (500 * 120) / 32 batches = 1875 batches
# 4 epochs = 4 * 30 records = (4 * 30) / 32 batches = 3.75 batches

# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 8 iterations of train data (epochs)
tf.logging.info("Before classifier.train")
classifier.train(
    input_fn=lambda: my_input_fn("movie_training.csv", 500, 256))
tf.logging.info("...done classifier.train")

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
tf.logging.info("Before classifier.evaluate")
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn("movie_test.csv", 11))
tf.logging.info("...done classifier.evaluate")
tf.logging.info("Evaluation results")
for key in evaluate_result:
    tf.logging.info("   {}, was: {}".format(key, evaluate_result[key]))

# Predict the type of some movie.
# Let's predict the examples in FILE_TEST, repeat only once.
predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, 1))
tf.logging.info("Prediction on test file")
for prediction in predict_results:
    # Will print the predicted class, i.e: 0, 1,2,3 or 4 if the prediction
    tf.logging.info("...{}".format(prediction["class_ids"]))

# Let create a dataset for prediction
# We've taken the first 1 examples in test file

prediction_input = [[217,2264909,141179,14024,419,95000000,5.9,9000,4.6,2079,1200],[0,0,987,1000,0,85000000,0,4000,6.0,1200,1500]]

def new_input_fn():
    def decode(x):
        x = tf.split(x, 11)  # Need to split into our  features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels

# Predict all our prediction_input
predict_results = classifier.predict(input_fn=new_input_fn)

# Print results
tf.logging.info("Predictions on memory")
for idx, prediction in enumerate(predict_results):
    type = prediction["\"]  # Get the predicted class (index)
    if type == 0:
        tf.logging.info("{} is Flop".format(prediction_input[idx]))
    elif type == 1:
        tf.logging.info("{}, is Average".format(prediction_input[idx]))
    elif type == 2:
        tf.logging.info("{}, is Hit".format(prediction_input[idx]))
    elif type == 3:
        tf.logging.info("{}, is Super Hit".format(prediction_input[idx]))
    else:
        tf.logging.info("{}, is Blockbuster".format(prediction_input[idx]))
