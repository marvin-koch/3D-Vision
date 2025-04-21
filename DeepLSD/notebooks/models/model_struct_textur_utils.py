import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def train_inductive(model, loader, optimizer, criterion,
                node_loss_w=1.0, device='cpu'):
    model.train()
    total_loss = 0
    batch_n = 0
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()

        
        node_pred = model(data)

        loss_node = criterion(node_pred, data.y)
        

        loss = node_loss_w * loss_node
        loss.backward()
        optimizer.step()
        # print(f"Loss for mini-batch {batch_n}: {loss.item()}")
        batch_n += 1
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_inductive(model, criterion, loader, device='cpu'):
    model.eval()
    node_loss_total = 0.0
    # Removed edge_loss_total
    total_nodes = 0
    # Removed total_edges

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            node_pred = model(data)

            # Ensure targets are float (for BCE-style losses)
            target_nodes = data.y.float()
            

            # Squeeze only if necessary to match dimensions
            if node_pred.shape != target_nodes.shape:
                node_pred = node_pred.squeeze()
            # Removed edge_pred squeeze check

            # Compute loss
            node_loss = criterion(node_pred, target_nodes)
            # Removed edge_loss calculation

            # Accumulate weighted loss
            node_loss_total += node_loss.item() * target_nodes.numel()
            # Removed edge_loss_total accumulation

            total_nodes += target_nodes.numel()
            # Removed total_edges accumulation

    avg_node_loss = node_loss_total / total_nodes if total_nodes > 0 else 0.0
    # Removed avg_edge_loss calculation

    return avg_node_loss # Adjusted return value


def test_inductive(model, loader, device='cpu', threshold_structural=0.5): # Removed threshold_coplanarity
    model.eval()

    correct_nodes = total_nodes = 0
    # Removed correct_edges, total_edges

    tp_nodes = fn_nodes = 0
    # Removed tp_edges, fn_edges

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            node_pred = model(data.x, data.edge_index)

            # Binary predictions
            pred_n = (node_pred >= threshold_structural).float()
            # Removed pred_e

            # Accuracy
            correct_nodes += (pred_n == data.y).sum().item()
            total_nodes += data.y.numel()

            # Removed edge accuracy calculation

            # Recall (TP / (TP + FN))
            tp_nodes += ((pred_n == 1) & (data.y == 1)).sum().item()
            fn_nodes += ((pred_n == 0) & (data.y == 1)).sum().item()

            # Removed edge recall calculation

    node_acc = correct_nodes / total_nodes if total_nodes > 0 else 0.0
    # Removed edge_acc calculation

    node_recall = tp_nodes / (tp_nodes + fn_nodes) if (tp_nodes + fn_nodes) > 0 else 0.0
    # Removed edge_recall calculation

    return node_acc, node_recall # Adjusted return value


def test_inductive_with_roc(model, loader, device='cpu'):
    model.eval()

    all_node_preds = []
    all_node_labels = []
    # Removed all_edge_preds, all_edge_labels

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            node_pred = model(data.x, data.edge_index)

            # Flatten and collect
            all_node_preds.append(node_pred.view(-1).cpu())
            all_node_labels.append(data.y.view(-1).float().cpu())

            # Removed edge pred/label collection

    # Concatenate all batches
    all_node_preds = torch.cat(all_node_preds).numpy()
    all_node_labels = torch.cat(all_node_labels).numpy()
    # Removed edge pred/label concatenation

    return all_node_preds, all_node_labels # Adjusted return value

def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
