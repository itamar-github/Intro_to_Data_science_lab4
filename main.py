from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score)


def question_1(points):
    """
    run 1-nn with the set of points. then classify the entire set and report accuracy
    :param points: list of Points
    :return: print accuracy
    """
    m = KNN(1)
    m.train(points)
    real = [point.label for point in points]
    predicted = m.predict(points)
    accuracy = accuracy_score(real, predicted)
    print(f"accuracy: {accuracy}")


def question_2(points):
    """
    run knn with 1<=k<=30. for each classifier run leave-one-out-cross-validation and get accuracy.
    :param points: List of Points
    :return: None
    """
    for i in range(1, 31):
        m = KNN(i)
        cv = CrossValidation()
        print(f"for k = {i}", end=' ')
        cv.run_cv(points, len(points), m, accuracy_score)


def question_3(points, k=19, fold_range=(2, 10, 20), print_final=False, print_folds=True):
    """
    run 19-nn with. for each number of folds for cross validation print fold accuracies.
    :param points: List of Points
    :param k: int for KNN
    :param fold_range: Tuple of values for the number of fold parameter for cross validation, for each k.
    :param print_final: prints the accuracy score for each k
    :param print_folds: prints the accuracy score of each fold for each k
    :return: print results
    """
    print("Question 3:")
    m = KNN(k)
    cv = CrossValidation()
    print(f"K={k}")
    # run cross validation with the different numbers of folds from fold_range
    for num_of_folds in fold_range:
        cv.run_cv(points, num_of_folds, m, accuracy_score, print_final, print_folds)


def question_4(points, k_range=(5, 7), n_folds=2):
    """
    run 2-fold-cross-validation for 5-NN and 7-NN and print accuracy score while:
    - not normalized
    - sum norm (L1 norm)
    - min-max norm
    - Z-norm
    :param points: List of Points
    :param k_range: tuple of k values
    :param n_folds: number of folds to run n-folds-cross-validation
    :return: None
    """
    print("Question 4:")

    # initialize normalizers
    dum_norm = DummyNormalizer()
    sum_norm = SumNormalizer()
    min_max_norm = MinMaxNormalizer()
    z_norm = ZNormalizer()
    normalizers = [dum_norm, sum_norm, min_max_norm, z_norm]
    # initialize cross validation object
    cv = CrossValidation()
    # first_line is used to adjust blank lines printing
    first_line = True
    # run the cross validation with all K in k_range
    for k in k_range:
        if not first_line:
            print()
        first_line = True
        print(f"K={k}")
        # initialize KNN with the current k
        m = KNN(k)
        # run the cross validation with the points normalized with the different normalizers
        for norm in normalizers:
            if not first_line:
                print()
            first_line = False
            # normalize
            norm.fit(points)
            new_points = norm.transform(points)
            # run cross validation
            norm_accuracy = cv.run_cv(new_points, n_folds, m, accuracy_score, False, True, False)
            print(f"Accuracy of {type(norm).__name__} is {norm_accuracy}")


if __name__ == '__main__':
    loaded_points = load_data()
    question_3(loaded_points)
    question_4(loaded_points)
