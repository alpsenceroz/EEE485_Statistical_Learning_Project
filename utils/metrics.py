import torch


def measure_metrics_fast(preds, labels):
    # ... existing imports ...
    
    stats = {}
    classes = sorted(list(set(labels.cpu().numpy())))  # Sort to ensure consistent ordering
    num_classes = len(classes)
    
    # Calculate accuracy using torch operations
    acc = (preds == labels).float().mean().item()
    
    # Create one-hot matrices for predictions and labels
    pred_one_hot = torch.zeros(len(preds), num_classes, device=preds.device)
    pred_one_hot.scatter_(1, preds.unsqueeze(1), 1)
    
    label_one_hot = torch.zeros(len(labels), num_classes, device=labels.device)
    label_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    # Calculate TP, FP, TN, FN for all classes at once
    tp = torch.sum(pred_one_hot * label_one_hot, dim=0)
    fp = torch.sum(pred_one_hot * (1 - label_one_hot), dim=0)
    fn = torch.sum((1 - pred_one_hot) * label_one_hot, dim=0)
    tn = torch.sum((1 - pred_one_hot) * (1 - label_one_hot), dim=0)
    
    # Calculate metrics for all classes at once
    prec = tp / (tp + fp)
    prec[torch.isnan(prec)] = 0  # Handle division by zero
    
    rec = tp / (tp + fn)
    rec[torch.isnan(rec)] = 0
    
    f1 = 2 * (prec * rec) / (prec + rec)
    f1[torch.isnan(f1)] = 0
    
    # Store individual class metrics
    for i, label in enumerate(classes):
        stats[label] = {
            "tp": tp[i].item(), "fp": fp[i].item(),
            "tn": tn[i].item(), "fn": fn[i].item(),
            "prec": prec[i].item(), "rec": rec[i].item(),
            "f1": f1[i].item()
        }
    
    # Calculate averages
    precs_avg = prec.mean().item()
    recs_avg = rec.mean().item()
    f1s_avg = f1.mean().item()
    
    print(f"Accuracy: {acc}, Precision: {precs_avg}, Recall: {recs_avg}, F1: {f1s_avg}")
    return acc, precs_avg, recs_avg, f1s_avg


def measure_metrics(preds, labels):
    stats = {}
    classes = set(labels)
    
    precs = 0
    recs = 0
    f1s = 0
    for label in classes:
        tp = torch.sum(torch.logical_and(preds == label, labels == label))
        fp = torch.sum(torch.logical_and(preds == label, labels != label))
        fn = torch.sum(torch.logical_and(preds != label, labels == label))
        tn = torch.sum(torch.logical_and(preds != label, labels != label))
        
        # tp = sum(1 for p, l in zip(preds, labels) if (p == label and l == label) )
        # fp = sum(1 for p, l in zip(preds, labels) if (p == label and l != label) )
        # tn = sum(1 for p, l in zip(preds, labels) if (p != label and l != label) )
        # fn = sum(1 for p, l in zip(preds, labels) if (p != label and l == label) )
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precs += prec  
        recs += rec
        f1s += f1
        
        stats[label] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "prec": prec, "rec": rec, "f1": f1}
        # print(stats[label])        
    
    precs /= len(classes)
    recs /= len(classes)
    f1s /= len(classes)
    
    acc = sum(1 for pred, label in zip(preds, labels) if pred == label) / len(preds)
    print(f"Accuracy: {acc}, Precision: {precs}, Recall: {recs}, F1: {f1s}")
    return acc, precs, recs, f1s

# def accuracy(preds, labels):
#     count = 0
#     for i in range(len(preds)):
#         if preds[i] == labels[i]:
#             count += 1
#     return count / len(preds)

# def precision(preds, labels):

#     classes = set(labels)
#     precisions = []
    

#     for c in classes:
#         tp = sum(1 for p, l in zip(preds, labels) if (p == c and l == c) )
#         tp_fp = sum(1 for p in preds if p == c)

#         class_precision = tp / tp_fp if tp_fp > 0 else 0
#         precisions.append(class_precision)
    
#     return sum(precisions) / len(precisions)

# def recall(preds, labels):
#     classes = set(labels)
#     recalls = []

#     for c in classes:
#         tp = sum(1 for p, l in zip(preds, labels) if (p == c and l == c) )
#         tp_fn = sum(1 for l in labels if l == c )

#         class_recall = tp / tp_fn if tp_fn > 0 else 0
#         recalls.append(class_recall)
    
#     return sum(recalls) / len(recalls)

# def f1_score(preds, labels):
#     classes = set(labels)
#     f1_scores = []
    
#     for c in classes:
#         tp = sum(1 for p, l in zip(preds, labels) if (p == c and l == c))
#         tp_fp = sum(1 for p in preds if p == c)
#         prec = tp / tp_fp if tp_fp > 0 else 0
        
#         tp_fn = sum(1 for l in labels if l == c)
#         rec = tp / tp_fn if tp_fn > 0 else 0
        
#         f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
#         f1_scores.append(f1)
    
#     return sum(f1_scores) / len(f1_scores)