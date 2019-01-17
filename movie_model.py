from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import movie_data
PATH = "/tmp/checkpoint_tf21/movie"


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = movie_data.read_csv_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    print("--> 2. Build 3 hidden layer DNN with 1024,512,256 units respectively.")
    my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Three hidden layers of 1024,512,256,128 nodes each.
        hidden_units=[1024,512,256,128],
        # The model must choose between 5 classes.
        n_classes=5,
        model_dir=PATH,
        config=my_checkpointing_config,
        optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.05,
        l1_regularization_strength=0.001,
    ))

    print("--> 3. Training the model using the test data.---> ")
    # Train the Model.
    classifier.train(
        input_fn=lambda:movie_data.create_dataset(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:movie_data.evaluate_input(test_x, test_y,
                                                args.batch_size))

   # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['SuperHit','Average','Hit']


    predict_x={
    'num_critic_for_reviews': [184,217,0],
    'revenue': [59600000,2264909,0],
    'num_voted_users': [116642,141179,987],
    'cast_total_facebook_likes': [1421,14024,1000],
    'num_user_for_reviews':[389,419,0],
    'budget': [950000,95000000,85000000],
    'imdb_score': [7.3,5.9,0],
    'movie_facebook_likes': [5000,9000,4000],
    'vote_average': [6.9,4.6,6.0],
    'vote_count': [953,2079,1200],
    'cast_power': [1194,1200,1500]
    }


    predictions = classifier.predict(
        input_fn=lambda:movie_data.evaluate_input(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    for pred_dict, expectation in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        print ("\nLabel Predicted:",class_id)
        probability = pred_dict['probabilities'][class_id]

        print(template.format(movie_data.CLASSIFICATION[class_id],
                              100 * probability, expectation))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)