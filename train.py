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
from models.cnn import LeNet5
from utils.metrics import measure_metrics, measure_metrics_fast


def get_config(args):
    # read the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config
    
def main(config):
    
    model_type = config["model_type"]
    results_dir = config["results_dir"] 
    
    # device agnostic code
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    
    if model_type == "BAYESIAN":
        # image must be flattened and contain integer valued features
        _, _, test_data, full_train_data = get_datasets(integer_valued=True, flatten=True)
        model = NaiveBayesianClassifier(num_features=28 * 28, num_labels=10)
        examples, labels = zip(*full_train_data)
        # model.fit_new(examples, labels)
        # model.save(Path(results_dir) / "bayesian_model.pkl")
        model.load(Path(results_dir) / "bayesian_model.pkl")
        test_examples, val_labels = zip(*test_data)
        preds = model.predict(test_examples, val_labels) # TODO: fix this
        
        test_preds_tensor = torch.tensor(preds)
        val_labels_tensor = torch.tensor(val_labels)
        # val_acc, val_prec, val_rec, val_f1 = measure_metrics(test_preds_tensor, val_labels_tensor)
        val_acc, val_prec, val_rec, val_f1 = measure_metrics_fast(test_preds_tensor, val_labels_tensor)
        
        print(f"Test accuracy: {val_acc:.4f}")
        print(f"Test precision: {val_prec:.4f}")
        print(f"Test recall: {val_rec:.4f}")
        print(f"Test f1: {val_f1:.4f}")
        
        
    elif model_type == "MLP" or model_type == "LeNet5":
        batch_size = config["batch_size"]
        lr = config["learning_rate"]
        
        
        
        if model_type == "MLP":
            train_data, val_data, _, _ = get_datasets(flatten=True)
            # examples, labels = zip(*train_data)
            # test_examples, test_labels = zip(*test_data)
        
            train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
            model = MLP
        elif model_type == "LeNet5":
            train_data, val_data, _, _ = get_datasets(flatten=False)
            # examples, labels = zip(*train_data)
            # test_examples, test_labels = zip(*test_data)
        
            train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
            val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
            model = LeNet5
        # model.to(device)
        loss = CrossEntropy()
        losses_epoch = []
        val_losses_epoch = []
        losses = []
        val_losses = []
        accs = []
        val_accs = []
        precs = []
        val_precs = []
        recs = []
        val_recs = []
        f1s = []
        val_f1s = []
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
            print("training_loss:", train_loss)
            
            val_predictions = []
            val_labels = []
            for batch in tqdm(val_dl):
                x, y = batch
                # x = x.to(device)
                # y = y.to(device)
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=10)
                val_preds_one_hot = model(x)
                val_loss_calc = loss(val_preds_one_hot, y_one_hot)
                
                val_losses_epoch.append(val_loss_calc.item())
                val_preds = torch.argmax(val_preds_one_hot, dim=1)
                val_predictions.extend(val_preds)
                val_labels.extend(y)
            
    
            val_loss = sum(val_losses_epoch) / len(val_losses_epoch)
            val_losses.append(val_loss)
            val_losses_epoch = []
            print("validation_loss:", val_loss)
            
            if (epoch) % 5 == 0:
                train_predictions_tensor = torch.tensor(train_predictions)
                train_labels_tensor = torch.tensor(train_labels)
                # acc, prec, rec, f1 = measure_metrics(train_predictions_tensor, train_labels_tensor)
                acc, prec, rec, f1 = measure_metrics_fast(train_predictions_tensor, train_labels_tensor)
                val_predictions_tensor = torch.tensor(val_predictions)
                val_labels_tensor = torch.tensor(val_labels)
                # val_acc, val_prec, val_rec, val_f1 = measure_metrics(val_predictions_tensor, val_labels_tensor)
                val_acc, val_prec, val_rec, val_f1 = measure_metrics_fast(val_predictions_tensor, val_labels_tensor)
                
                accs.append(acc)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                
                
                val_accs.append(val_acc)
                val_precs.append(val_prec)
                val_recs.append(val_rec)
                val_f1s.append(val_f1)
            
        # save the model
        save_dir = Path(results_dir) / f"{model_type.lower()}_batch_size_{config['batch_size']}_lr_{config['learning_rate']}"
        save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
        model.save(save_dir /f"{model_type.lower()}-{config['batch_size']}-{config['learning_rate']}.pkl")
        
        # plot the metrics
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])  # This makes ax5 span both columns in the last row
        
        fig.suptitle(f'{model_type} Training Metrics', fontsize=16, y=1.02)
        
        # Common styling function
        def style_plot(ax, title, xlabel, ylabel, ylim=None):
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_title(title, pad=10, fontsize=12)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.legend()

        # Plot accuracy
        ax1.plot(accs, label='Train')
        ax1.plot(val_accs, color='red', label='Validation')
        style_plot(ax1, 'Accuracy', 'Epoch', 'Accuracy', (0, 1))

        # Plot precision
        ax2.plot(precs, label='Train')
        ax2.plot(val_precs, color='red', label='Validation')
        style_plot(ax2, 'Precision', 'Epoch', 'Precision', (0, 1))

        # Plot recall
        ax3.plot(recs, label='Train')
        ax3.plot(val_recs, color='red', label='Validation')
        style_plot(ax3, 'Recall', 'Epoch', 'Recall', (0, 1))

        # Plot F1 Score
        ax4.plot(f1s, label='Train')
        ax4.plot(val_f1s, color='red', label='Validation')
        style_plot(ax4, 'F1 Score', 'Epoch', 'F1 Score', (0, 1))

        # Plot losses
        ax5.plot(losses, label='Train')
        ax5.plot(val_losses, color='red', label='Validation')
        style_plot(ax5, 'Loss', 'Epoch', 'Loss')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_dir / f"{model_type.lower()}-{config['batch_size']}-{config['learning_rate']}.png", 
                   bbox_inches='tight', dpi=300)
        
        print("Plot saved.")
        # plt.show()
        
    # elif model_type == "CNN":
    #     batch_size = config["batch_size"]
    #     train_data, test_data = get_datasets()
    #     train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    #     test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    #     pass
    else:
        raise ValueError(f"Model type {model_type} not supported")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/mlp_config.yaml")
    parser.add_argument("--hyperparameter-test", action="store_true", 
                        help="Run hyperparameter testing across multiple learning rates and batch sizes")
    
    args = parser.parse_args()
    config = get_config(args)
    
    if args.hyperparameter_test:
        lrs = [0.03, 0.003, 0.0003]
        # batch_sizes = [8, 16, 32, 64]
        batch_sizes = [32, 16]
        for batch_size in batch_sizes:
            for lr in lrs:
                config["batch_size"] = batch_size
                config["learning_rate"] = lr
                main(config)
    else:
        main(config)
