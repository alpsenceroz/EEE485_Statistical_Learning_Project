import torch

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
    print(f"Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1}")
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