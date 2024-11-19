
import math
class NaiveBayesianClassifier:
    # """_summary_
    # This is a naive bayesian classifier that uses the naive bayesian algorithm to classify data.
    # """
    def __init__(self, num_features:int, num_labels:int, smoothing_factor:float=1.0):
        """_summary_

        Args:
            num_features (int): number of features
            num_labels (int): number of labels
            smoothing_factor (float): smoothing factor for the model
        """
        self.label_probs = {}  # P(y)
        self.feature_probs = {}  # P(X[i] | C)
        self.num_features = num_features
        self.num_labels = num_labels
        self.smoothing_factor = smoothing_factor
        self.num_unique_features = {}
        self.num_examples = 0


    def fit(self, X_train, y_train):
        """fits model to given dataset usign naive bayesian algorithm

        Args:
            X_train: array of training examples, each example has num_features features
            y_train: training labels
        """
        # reset probs for new data
        for label in range(self.num_labels):
            self.label_probs[label] = 0
            self.feature_probs[label] = {}

        self.num_examples  = len(X_train)
        
        # TOOD: optimize the loops
        # calculate number of unique features for each feature index

        for feature_index in range(len(X_train[0])):
            features = [example[feature_index] for example in X_train]
            self.num_unique_features[feature_index] = len(set(features))

        for label in range(self.num_features):
            print(f"calculating for label: {label}")
            # calculate P(Y) for each class
            class_examples = [ X_train[i] for i in range(self.num_examples) if y_train[i] == label ]
            self.label_probs[label] = len(class_examples) / self.num_examples 
            
            # calculate feature probability for given class c = P(xi | Y) with Laplace smoothing
            for feature_index in range(len(X_train[0])):
                print(f'calculatin for feature: {feature_index}')
                features = [example[feature_index] for example in class_examples]
                self.feature_probs[label][feature_index] = {}

                for feature in set(features):
                    self.feature_probs[label][feature_index][feature] = \
                        (len([f for f in features if f == feature]) + self.smoothing_factor) \
                        / (self.num_examples + self.smoothing_factor * self.num_unique_features[feature_index])


    def predict(self, X_test):
        """predicts labels for given list of examples

        Args:
            X_test: 2D list of examples

        Returns:
            list: list of predicted labels
        """
        res = []
        for ex_index, example in enumerate(X_test):
            pred = None
            max_prob = 0
            for label in self.label_probs.keys():
                
                if (label not in self.label_probs.keys()):
                    prob = 0
                    break
                
                prob = self.label_probs[label]
                
                for feature_index, feature in enumerate(example):
                    if (feature_index not in self.feature_probs[label].keys()):
                        raise ValueError(f"Feature index {feature_index} not in feature probabilities for label {label}")
                    elif (feature not in self.feature_probs[label][feature_index].keys()):
                        # print(
                        #     f"0 probability at index {ex_index} of class {label} for feature {feature_index} and value {feature}"
                        # )
                        prob = prob + math.log(self.smoothing_factor / (self.num_unique_features[feature_index] * self.smoothing_factor + self.num_examples))
                        break
                    else:
                        prob = prob + math.log(self.feature_probs[label][feature_index][feature])

                if prob > max_prob:
                    max_prob = prob
                    pred = label
            if pred is None:
                # print(f"0 probability for example {ex_index}")
                pred = "POSITIVE" # for zero frequency values, assume test results are positive
                pass
            res.append(pred)
        return res
