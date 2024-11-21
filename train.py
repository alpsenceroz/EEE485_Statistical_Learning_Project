import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.loader import get_datasets
from utils.losses import CrossEntropy, MSE
from models.bayesian import NaiveBayesianClassifier
from models.mlp import MLP
from utils.metrics import accuracy, precision, recall, f1_score, measure_metrics

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
        
        test_preds_tensor = torch.tensor(preds)
        test_labels_tensor = torch.tensor(test_labels)
        test_acc, test_prec, test_rec, test_f1 = measure_metrics(test_preds_tensor, test_labels_tensor)
        
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test precision: {test_prec:.4f}")
        print(f"Test recall: {test_rec:.4f}")
        print(f"Test f1: {test_f1:.4f}")
        
        
    elif model_type == "MLP":
        batch_size = config["batch_size"]
        lr = config["learning_rate"]
        
        train_data, test_data = get_datasets(flatten=True)
        # examples, labels = zip(*train_data)
        # test_examples, test_labels = zip(*test_data)
    
        train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        model = MLP
        # model.to(device)
        loss = CrossEntropy()
        losses_epoch = []
        test_losses_epoch = []
        losses = []
        test_losses = []
        accs = []
        test_accs = []
        precs = []
        test_precs = []
        recs = []
        test_recs = []
        f1s = []
        test_f1s = []
        for epoch in range(config["epochs"]):
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            train_predictions = []
            train_labels = []
            
            for batch in tqdm(train_dl):
                x, y = batch
                # x = x.to(device)
                # y = y.to(device)
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=10)
                preds_one_hot = model(x)
                loss_calc = loss(preds_one_hot, y_one_hot)
                dE_do = loss.backward()
                model.backward(dE_do, lr)
                
                losses_epoch.append(loss_calc.item())
                preds = torch.argmax(preds_one_hot, dim=1)
                train_predictions.extend(preds)
                train_labels.extend(y)
            train_loss = sum(losses_epoch) / len(losses_epoch)
            losses.append(train_loss)
            losses_epoch = []
            print(train_loss)
            
            test_predictions = []
            test_labels = []
            for batch in tqdm(test_dl):
                x, y = batch
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=10)
                test_preds_one_hot = model(x)
                test_loss_calc = loss(test_preds_one_hot, y_one_hot)
                
                test_losses_epoch.append(test_loss_calc.item())
                test_preds = torch.argmax(test_preds_one_hot, dim=1)
                test_predictions.extend(test_preds)
                test_labels.extend(y)
            
    
            test_loss = sum(test_losses_epoch) / len(test_losses_epoch)
            test_losses.append(test_loss)
            test_losses_epoch = []
            print(test_loss)
            
            if (epoch + 1) % 10 == 0:
                # train_predictions_tensor = torch.tensor(train_predictions)
                # train_labels_tensor = torch.tensor(train_labels)
                # acc, prec, rec, f1 = measure_metrics(train_predictions_tensor, train_labels_tensor)
                test_predictions_tensor = torch.tensor(test_predictions)
                test_labels_tensor = torch.tensor(test_labels)
                test_acc, test_prec, test_rec, test_f1 = measure_metrics(test_predictions_tensor, test_labels_tensor)
                
                # accs.append(acc)
                # precs.append(prec)
                # recs.append(rec)
                # f1s.append(f1)
                
                
                test_accs.append(test_acc)
                test_precs.append(test_prec)
                test_recs.append(test_rec)
                test_f1s.append(test_f1)
            
                        
        # plot the metrics
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 8))
        
        # ax1.plot(accs)
        ax1.plot(test_accs, color='red')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # ax2.plot(precs)
        ax2.plot(test_precs, color='red')
        ax2.set_title('Precision')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)

        # ax3.plot(recs)
        ax3.plot(test_recs, color='red')
        ax3.set_title('Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        
        # ax4.plot(f1s)
        ax4.plot(test_f1s, color='red')
        ax4.set_title('F1 Score')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1 Score')
        ax4.set_ylim(0, 1)
        
        ax5.plot(losses)
        ax5.plot(test_losses, color='red')
        ax5.set_title('Losses')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
        
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
