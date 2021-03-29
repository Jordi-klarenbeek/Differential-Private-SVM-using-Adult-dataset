import huber_SVM
import time
import numpy as np
from dataset_processing import preprocessing
from dataset_processing import buckets_generator

""""
Run tests for different values of epsilon
"""
def run_epsilon(data, buckets_loc):
    #test_epsilon = [0.0000000001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    test_epsilon = [0.001, 0.05, 1.0]
    labda = 10**-2.5
    results_run = np.zeros(4)

    """"
    Enter amount of run times
    """
    times = 100

    for j in range(0,times):
        buckets_loc = buckets_generator.generate_random_index_buckets(data)
        print("[Run number: {}]".format(j))

        results_buckets = np.zeros(4)
        for i in range(0,10):
            # Retrieve the training and testing data according to a specific bucket number (starts from 0)
            training_data, testing_data = preprocessing.cross_validation_train_test_split(data, buckets_location=buckets_loc, bucket_number=i)

            print("[Bucket number: {}]".format(i))

            x = 0
            for eps in test_epsilon:
                t0 = time.time()

                # create, train and test SVM
                huber = huber_SVM.SVM(private=True, labda=labda, h=0.5)
                huber.fit(training_data, epsilon_p=eps)
                eval = 1 - huber.evaluate(testing_data)

                # compute runtime for runtime analysis
                t1 = time.time() - t0
                results_buckets[x] = results_buckets[x] + eval
                x = x + 1
                print('{}, {}'.format(eval, t1))

            t0 = time.time()
            labda_nonpriv = 10 ** -2.5
            huber_nonpriv = huber_SVM.SVM(private=False, labda=labda_nonpriv, h=0.5)
            huber_nonpriv.fit(training_data, epsilon_p=None)
            eval_nonpriv = 1 - huber_nonpriv.evaluate(testing_data)
            t1 = time.time() - t0

            results_buckets[3] = results_buckets[3] + eval_nonpriv
            print('{}'.format(eval_nonpriv))

        # Calculate and print the average of a single run, from the 10 fold cross-validation
        average_buckets = results_buckets/10
        results_run = results_run + average_buckets
        print(results_run)

    # Calculate and print the total average of all runs
    average_run = results_run/times
    print(average_run)

""""
Run tests for different values of lambda
"""
def run_lambda(data, buckets_loc):
    test_lambda = [10**-10.0, 10**-7.0, 10**-5.0, 10**-3.5, 10**-3.0, 10**-2.5, 10**-2.0, 10**-1.5]

    for i in range(10):
        # Retrieve the training and testing data according to a specific bucket number (starts from 0)
        training_data, testing_data = preprocessing.cross_validation_train_test_split(data, buckets_location=buckets_loc, bucket_number=i)
        print("[Bucket number: {}]".format(i))

        for labda in test_lambda:
            # create, train and test SVM
            huber = huber_SVM.SVM(private=True, labda=labda, h=0.5)
            huber.fit(training_data, epsilon_p=0.1)
            eval_priv = 1 - huber.evaluate(testing_data)

            print('[Accuracy: {}, lambda: {}, private]'.format(eval_priv, labda))

            # create, train and test SVM
            huber_nonpriv = huber_SVM.SVM(private=False, labda=labda, h=0.5)
            huber_nonpriv.fit(training_data, epsilon_p=None)
            eval_nonpriv = 1 - huber_nonpriv.evaluate(testing_data)

            print('[Accuracy: {}, lambda: {}, non-private]'.format(eval_nonpriv, labda))

""""
Run tests for different values of h
"""
def run_h(data, buckets_loc):
    test_h = [0.01, 0.05, 0.1, 0.5]
    labda = 10**-2.5

    for i in range(10):
        # Retrieve the training and testing data according to a specific bucket number (starts from 0)
        training_data, testing_data = preprocessing.cross_validation_train_test_split(data, buckets_location=buckets_loc, bucket_number=i)

        print("[Bucket number: {}]".format(i))

        for h in test_h:
            # create, train and test SVM
            huber = huber_SVM.SVM(private=True, labda=labda, h=h)
            huber.fit(training_data, epsilon_p=0.1)
            eval_priv = 1 - huber.evaluate(testing_data)

            print()
            print('[Accuracy: {}, h: {}, private]'.format(eval_priv, h))

            # create, train and test SVM
            huber_nonpriv = huber_SVM.SVM(private=False, labda=labda, h=h)
            huber_nonpriv.fit(training_data, epsilon_p=None)
            eval_nonpriv = 1 - huber_nonpriv.evaluate(testing_data)

            print('[Accuracy: {}, h: {}, non-private]'.format(eval_nonpriv, h))
            print()