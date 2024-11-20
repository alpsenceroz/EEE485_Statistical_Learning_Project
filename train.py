import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.loader import get_datasets
from utils.losses import CrossEntropy, MSE
from models.bayesian import NaiveBayesianClassifier

from models.mlp import MLP



def main(args):
    # read the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model_type = config["model_type"]
    results_dir = config["results_dir"] 
    
    # device agnostic code
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    
    if model_type == "BAYESIAN":
        # image must be flattened and contain integer valued features
        train_data, test_data = get_datasets(integer_valued=True, flatten=True)
        model = NaiveBayesianClassifier(num_features=28 * 28, num_labels=10)
        examples, labels = zip(*train_data)
        # model.fit_new(examples, labels)
        # model.save(Path(results_dir) / "bayesian_model.pkl")
        model.load(Path(results_dir) / "bayesian_model.pkl")
        test_examples, test_labels = zip(*test_data)
        preds = model.predict(test_examples, test_labels) # TODO: fix this
        count = 0
        for i in range(len(preds)):
            if preds[i] == test_labels[i]:
                count += 1
        accuracy = count / len(preds)
        print(f"Test accuracy: {accuracy:.4f}")
        
        pass
    elif model_type == "MLP":
        batch_size = config["batch_size"]
        lr = config["learning_rate"]
        train_data, test_data = get_datasets(flatten=True)
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
        model = MLP
        loss = CrossEntropy
        for epoch in range(config["epochs"]):
            for batch in train_dl:
                x, y = batch
                preds = model(x)
                loss_calc = loss(preds, y)
                dE_do = loss.backward()
                print("dE_do.shape", dE_do.shape)
                model.backward(dE_do, lr)
                print(loss_calc)
    elif model_type == "CNN":
        batch_size = config["batch_size"]
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
