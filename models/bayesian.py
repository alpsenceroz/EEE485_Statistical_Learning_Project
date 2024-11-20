
import math
import pickle
from tqdm import tqdm
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


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    

    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict.__dict__)


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
                        / (len(features) + self.smoothing_factor * self.num_unique_features[feature_index])


    def fit_new(self, X_train, y_train):
        feature_counts = {}
        feature_unique_lists = {}
        self.class_counts = {}
        
        # iterate over all examples and get statistics
        print(f"Getting statistics")
        for example, label in tqdm(zip(X_train, y_train), total=len(X_train)):
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            for feature_index, feature in enumerate(example):
                feature = feature.item()
                feature_counts[label] = feature_counts.get(label, {})
                feature_counts[label][feature_index] = feature_counts[label].get(feature_index, {})
                feature_counts[label][feature_index][feature] = feature_counts[label][feature_index].get(feature, 0) + 1
                feature_unique_lists[feature_index] = feature_unique_lists.get(feature_index, [])
                if feature not in feature_unique_lists[feature_index]:
                    feature_unique_lists[feature_index].append(feature)
                
        # calculate unique number of features for the Laplacian smoothing
        print(f"Calculating number of unique features")
        for feature_index in range(self.num_features):
            self.num_unique_features[feature_index] = len(feature_unique_lists.get(feature_index, []))
            
        # calculate probabilities
        print(f"Calculating probabilities")
        for label in self.class_counts.keys():
            print(f"Calculating for label: {label}")
            self.label_probs[label] = self.class_counts[label] / len(X_train)
            for feature_index in range(self.num_features):
                print(f"Calculating for feature index: {feature_index}")
                self.feature_probs[label] = self.feature_probs.get(label, {})
                self.feature_probs[label][feature_index] = self.feature_probs[label].get(feature_index, {})
                for feature, count in feature_counts[label][feature_index].items():
                    # print(f"Calculating for feature: {feature}")
                    # print(f"Count: {count}")
                    self.feature_probs[label][feature_index][feature] = (count + self.smoothing_factor) / (self.class_counts[label] + self.smoothing_factor * self.num_unique_features[feature_index])
        
        
        # print(self.feature_probs.keys())
        # for label in self.feature_probs.keys():
        #     print(label,self.feature_probs[label])

    def predict(self, X_test, y_test):
        """predicts labels for given list of examples

        Args:
            X_test: 2D list of examples

        Returns:
            list: list of predicted labels
        """
        res = []
        for example, label in tqdm(zip(X_test, y_test), total=len(X_test)):
            pred = None
            max_prob = float('-inf')
            for label in self.label_probs.keys():
                
                if (label not in self.label_probs.keys()):
                    prob = 0
                    print(f"Label {label} not in label probabilities")
                    break
                
                prob = self.label_probs[label]
                
                for feature_index, feature in enumerate(example):
                    feature = feature.item()
                    if (feature_index not in self.feature_probs[label].keys()):
                        raise ValueError(f"Feature index {feature_index} not in feature probabilities for label {label}")
                    elif (feature not in self.feature_probs[label][feature_index].keys()):
                        prob = prob + math.log(self.smoothing_factor / (self.num_unique_features[feature_index] * self.smoothing_factor + self.class_counts[label]))
                    else:
                        prob = prob + math.log(self.feature_probs[label][feature_index][feature])

                if prob > max_prob:
                    max_prob = prob
                    pred = label
            if pred is None:
                # print(f"0 probability for example {ex_index}")
                pred = 0# for zero frequency values, assume test results are zero
                pass
            res.append(pred)
        return res
