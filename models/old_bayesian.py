import sys


class NaiveBayesianClassifier:
    def __init__(self, column_names):
        self.label_probs = {}  # P(y)
        self.feature_probs = {}  # P(X[i] | C)
        self.column_names = column_names

    def fit(self, X_train, y_train):
        """fits model to given dataset usign naive bayesian algorithm

        Args:
            X_train (list): 2D training examples
            y_train (list): 1D training labels
        """
        # reset probs for new data
        self.label_probs = {"POSITIVE": 0, "NEGATIVE": 0}
        self.feature_probs = {
            "POSITIVE": {},
            "NEGATIVE": {},
        }  # 3 dimentional dictionary

        sample_size = len(X_train)

        for label in {"NEGATIVE", "POSITIVE"}:
            # calculate P(Y) for each class
            # self.feature_probs[label] = {}
            class_examples = [
                X_train[i] for i in range(sample_size) if y_train[i] == label
            ]
            self.label_probs[label] = len(class_examples) / sample_size
            # calculate feature probability for given class c = P(xi | Y)
            for feature_index in range(len(X_train[0])):
                features = [example[feature_index] for example in class_examples]
                self.feature_probs[label][feature_index] = {}

                for feature in set(features):
                    self.feature_probs[label][feature_index][feature] = len(
                        [f for f in features if f == feature]
                    ) / len(features)

    def predict(self, X_test):
        """predicts labels for given list of examples

        Args:
            X_test (list): 2D list of examples

        Returns:
            list: list of predicted labels
        """
        res = []
        for ex_index, example in enumerate(X_test):
            pred = None
            max_prob = 0
            for label in self.label_probs.keys():
                prob = self.label_probs[label]
                for feature_index, feature in enumerate(example):
                    if (
                        label not in self.feature_probs.keys()
                        or feature_index not in self.feature_probs[label].keys()
                        or feature
                        not in self.feature_probs[label][feature_index].keys()
                    ):
                        # print(
                        #     f"0 probability at index {ex_index} of class {label} for feature {feature_index} and value {feature}"
                        # )
                        prob = 0
                        break
                    else:
                        prob = prob * self.feature_probs[label][feature_index][feature]

                if prob > max_prob:
                    max_prob = prob
                    pred = label
            if pred is None:
                # print(f"0 probability for example {ex_index}")
                pred = "POSITIVE" # for zero frequency values, assume test results are positive
                pass
            res.append(pred)
        return res

    def test(self, X_test, y_test):
        """tests given data and calculates metrics

        Args:
            X_train (list): 2D test examples
            y_train (list): 1D test labels

        Returns:
            dict: dictionary containing confusion matrix, accuracy, correct and incorrect number of examples
        """
        metrics = {}
        preds = self.predict(X_test)
        tp = len(
            [
                i
                for i in range(len(preds))
                if preds[i] == "POSITIVE" and y_test[i] == "POSITIVE"
            ]
        )
        fp = len(
            [
                i
                for i in range(len(preds))
                if preds[i] == "POSITIVE" and y_test[i] != "POSITIVE"
            ]
        )
        tn = len(
            [
                i
                for i in range(len(preds))
                if preds[i] == "NEGATIVE" and y_test[i] == "NEGATIVE"
            ]
        )
        fn = len(
            [
                i
                for i in range(len(preds))
                if preds[i] == "NEGATIVE" and y_test[i] != "NEGATIVE"
            ]
        )

        metrics["tp"] = tp
        metrics["fp"] = fp
        metrics["tn"] = tn
        metrics["fn"] = fn
        metrics["accuracy"] = (tp + tn) / len(preds)
        metrics["correct"] = tp + tn
        metrics["incorrect"] = fp + fn

        return metrics

    def find_most_least_effective(self, label):
        """find the most and least effective features for given label

        Args:
            label (string): label name

        Returns:
            tuple: 2D tuple with two elements. First element contains max feature category and its value, second  element contains min feature category and its value
        """
        max_val = 0
        max_feature = ""
        max_feature_in = 0
        min_val = sys.maxsize
        min_feature = ""
        min_feature_in = 0
        for feature_in, feature_dict in self.feature_probs[label].items():
            for feature, value in feature_dict.items():
                if value > max_val:
                    max_val = value
                    max_feature = feature
                    max_feature_in = feature_in
                if value < min_val:
                    min_val = value
                    min_feature = feature
                    min_feature_in = feature_in
        return (
            (self.column_names[max_feature_in], max_feature),
            (self.column_names[min_feature_in], min_feature),
        )


def read_csv(path, y=False):
    f = open(path, "r")
    content = f.readlines()
    if y:
        content = [line.strip() for line in content]
    else:
        content = [line.strip().split(",") for line in content]
    col_names = content[0]
    rows = content[1:]
    return col_names, rows


def main():
    # read data from dataset
    X_TRAIN_PATH = "Q3/data_x_train.csv"
    Y_TRAIN_PATH = "Q3/data_y_train.csv"
    X_TEST_PATH = "Q3/data_x_test.csv"
    Y_TEST_PATH = "Q3/data_y_test.csv"
    col_names, X_train = read_csv(X_TRAIN_PATH)
    _, y_train = read_csv(Y_TRAIN_PATH, y=True)
    _, X_test = read_csv(X_TEST_PATH)
    _, y_test = read_csv(Y_TEST_PATH, y=True)

    # initialize naive bayesian classifier
    classifier = NaiveBayesianClassifier(col_names)

    # fit to data
    classifier.fit(X_train, y_train)
    metrics = classifier.test(X_test, y_test)
    print(metrics)
    max_p_feature, min_p_feature = classifier.find_most_least_effective("POSITIVE")
    print(
        f"The most effective feature for POSITIVE label is {max_p_feature[0]} with value {max_p_feature[1]}"
    )
    print(
        f"The least effective feature for POSITIVE label is {min_p_feature[0]} with value {min_p_feature[1]}"
    )


if __name__ == "__main__":
    main()
