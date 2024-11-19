import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from utils.loader import get_datasets
from models.bayesian import NaiveBayesianClassifier
# from models.mlp import MLP
# from models.cnn import CNN



def main(args):
    # read the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    
    
    model_type = config["model_type"]
    batch_size = config["batch_size"]
    

    
    if model_type == "BAYESIAN":
        # image must be flattened and contain integer valued features
        train_data, test_data = get_datasets(integer_valued=True, flatten=True)
        model = NaiveBayesianClassifier(num_features=28 * 28, num_labels=10)
        examples, labels = zip(*train_data)
        print(examples[0])
        print(labels[0])
        model.fit(examples, labels)
        preds = model.predict(test_data)
        print(preds[0])
        pass
    elif model_type == "MLP":
        train_data, test_data = get_datasets()
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
        pass
    elif model_type == "CNN":
        train_data, test_data = get_datasets()
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
        pass
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    
    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/mlp_config.yaml")
    args = parser.parse_args()
    main(args)
