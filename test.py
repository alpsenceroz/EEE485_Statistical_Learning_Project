import argparse
import yaml
from pathlib import Path
import torch
from tqdm import tqdm
from utils.metrics import measure_metrics_fast,measure_metrics
from utils.loader import get_datasets
from models.bayesian import NaiveBayesianClassifier
from models.mlp import MLP
from models.cnn import LeNet5


def load_model(model_type, model_weights):
    """Load the appropriate model based on the model_type."""
    if model_type == "BAYESIAN":
        model = NaiveBayesianClassifier(num_features=28 * 28, num_labels=10)
        model.load(model_weights)
    elif model_type == "MLP":
        model = MLP
        model.load(model_weights)
    elif model_type == "CNN":
        model = LeNet5
        model.load(model_weights)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def evaluate_model(model, dataloader, is_bayesian=False):
    """Evaluate the model and calculate metrics."""
    all_preds, all_labels = [], []
    for x, y in tqdm(dataloader, desc="Evaluating"):
        if is_bayesian:
            preds = model.predict(x, y)
            preds = torch.tensor(preds)
        else:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(y.cpu())
    
    preds_tensor = torch.tensor(all_preds)
    labels_tensor = torch.tensor(all_labels)
    return measure_metrics_fast(preds_tensor, labels_tensor)


def main(args):
    # Load configurations
    model_type = args.type

    batch_size = 64

    # Determine device
    # device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'

    # Get datasets and dataloaders
    if model_type == "BAYESIAN":
        _, _, test_data, _ = get_datasets(integer_valued=True, flatten=True)
        is_bayesian = True
    else:
        _, _, test_data, _ = get_datasets(flatten=(model_type == "MLP"))
        is_bayesian = False

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_model(model_type, args.model)

    # Evaluate and measure metrics
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, is_bayesian)

    # Output metrics
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision: {test_prec:.4f}")
    print(f"Test recall: {test_rec:.4f}")
    print(f"Test f1: {test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights.")
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["MLP", "CNN", "BAYESIAN"],
        help="Model type"
    )
    args = parser.parse_args()


    # Execute main function
    main(args)