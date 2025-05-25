import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#****************************************************************
#****************************************************************
#****************************************************************
#                       Inductive

def train_inductive(model, loader, optimizer, criterion, 
                node_loss_w=1.0, edge_loss_w=1.0, device='cpu'):
    model.train()
    total_loss = 0
    batch_n = 0
    for data in loader:

        data = data.to(device)
        optimizer.zero_grad()

        node_pred, edge_pred = model(data.x, data.edge_index)

        loss_node = criterion(node_pred, data.y)
        loss_edge = criterion(edge_pred, data.edge_labels)

        loss = node_loss_w * loss_node + edge_loss_w * loss_edge
        loss.backward()
        optimizer.step()
        # print(f"Loss for mini-batch {batch_n}: {loss.item()}")
        batch_n += 1
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_inductive(model, criterion, loader, device='cpu'):
    model.eval()
    node_loss_total = 0.0
    edge_loss_total = 0.0
    total_nodes = 0
    total_edges = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_pred, edge_pred = model(data.x, data.edge_index)

            # Ensure targets are float (for BCE-style losses)
            target_nodes = data.y.float()
            target_edges = data.edge_labels.float()

            # Squeeze only if necessary to match dimensions
            if node_pred.shape != target_nodes.shape:
                node_pred = node_pred.squeeze()
            if edge_pred.shape != target_edges.shape:
                edge_pred = edge_pred.squeeze()

            # Compute loss
            node_loss = criterion(node_pred, target_nodes)
            edge_loss = criterion(edge_pred, target_edges)

            # Accumulate weighted loss
            node_loss_total += node_loss.item() * target_nodes.numel()
            edge_loss_total += edge_loss.item() * target_edges.numel()

            total_nodes += target_nodes.numel()
            total_edges += target_edges.numel()

    avg_node_loss = node_loss_total / total_nodes if total_nodes > 0 else 0.0
    avg_edge_loss = edge_loss_total / total_edges if total_edges > 0 else 0.0

    return avg_node_loss, avg_edge_loss


def test_inductive(model, loader, device='cpu', threshold_structural=0.5, threshold_coplanarity=0.5):
    model.eval()

    correct_nodes = total_nodes = 0
    correct_edges = total_edges = 0

    tp_nodes = fn_nodes = 0
    tp_edges = fn_edges = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_pred, edge_pred = model(data.x, data.edge_index)

            # Binary predictions
            pred_n = (node_pred >= threshold_structural).float()
            pred_e = (edge_pred >= threshold_coplanarity).float()

            # Accuracy
            correct_nodes += (pred_n == data.y).sum().item()
            total_nodes += data.y.numel()

            correct_edges += (pred_e == data.edge_labels).sum().item()
            total_edges += data.edge_labels.numel()

            # Recall (TP / (TP + FN))
            tp_nodes += ((pred_n == 1) & (data.y == 1)).sum().item()
            fn_nodes += ((pred_n == 0) & (data.y == 1)).sum().item()

            tp_edges += ((pred_e == 1) & (data.edge_labels == 1)).sum().item()
            fn_edges += ((pred_e == 0) & (data.edge_labels == 1)).sum().item()

    node_acc = correct_nodes / total_nodes if total_nodes > 0 else 0.0
    edge_acc = correct_edges / total_edges if total_edges > 0 else 0.0

    node_recall = tp_nodes / (tp_nodes + fn_nodes) if (tp_nodes + fn_nodes) > 0 else 0.0
    edge_recall = tp_edges / (tp_edges + fn_edges) if (tp_edges + fn_edges) > 0 else 0.0

    return node_acc, edge_acc, node_recall, edge_recall


def test_inductive_with_roc(model, loader, device='cpu'):
    model.eval()

    all_node_preds = []
    all_node_labels = []
    all_edge_preds = []
    all_edge_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            node_pred, edge_pred = model(data.x, data.edge_index)

            # Flatten and collect
            all_node_preds.append(node_pred.view(-1).cpu())
            all_node_labels.append(data.y.view(-1).float().cpu())

            all_edge_preds.append(edge_pred.view(-1).cpu())
            all_edge_labels.append(data.edge_labels.view(-1).float().cpu())

    # Concatenate all batches
    all_node_preds = torch.cat(all_node_preds).numpy()
    all_node_labels = torch.cat(all_node_labels).numpy()
    all_edge_preds = torch.cat(all_edge_preds).numpy()
    all_edge_labels = torch.cat(all_edge_labels).numpy()

    return all_node_preds, all_node_labels, all_edge_preds, all_edge_labels

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



#****************************************************************
#****************************************************************
#****************************************************************
#                       Transductive

def train(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0
    for sub_data in loader:
        optimizer.zero_grad()
        
        out = model(sub_data.x, sub_data.edge_index)
        
        # y shape should match [num_nodes, 1]
        target = sub_data.y.unsqueeze(1).float()
        
        loss = criterion(out[sub_data.train_mask], target[sub_data.train_mask])
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss


def test(model, criterion, loader, threshold=0.5):
    model.eval()
    train_accs, test_accs = [], [], []
    with torch.no_grad():
        for sub_data in loader:
            out = model(sub_data.x, sub_data.edge_index)
            
            preds = (out >= threshold).float()
            target = sub_data.y.unsqueeze(1).float()

            tuple_mask_accuracylist =  zip([sub_data.train_mask, sub_data.test_mask], [train_accs, test_accs])
            for mask, acc_list in tuple_mask_accuracylist:
                correct = (preds[mask] == target[mask]).float().sum()
                accuracy = correct / mask.sum().float()
                acc_list.append(accuracy.item())
    avg_train_acc = sum(train_accs) / len(train_accs)
    avg_test_acc  = sum(test_accs) / len(test_accs)
    return avg_train_acc, avg_test_acc


#****************************************************************


def train_combined(model, optimizer, criterion, loader, node_loss_weight=1.0, edge_loss_weight=1.0):
    model.train()
    total_loss = 0
    for sub_data in loader:
        optimizer.zero_grad()
        
        node_out, edge_out = model(sub_data.x, sub_data.edge_index)
        
        # Node-level Loss Computation
        node_target = sub_data.y.unsqueeze(1).float()
        loss_node = criterion(node_out[sub_data.train_mask], node_target[sub_data.train_mask])
        
        # Edge-level Loss Computation
        edge_target = sub_data.edge_labels.unsqueeze(1).float()
        loss_edge = criterion(edge_out, edge_target)
        
        # Combine
        loss = node_loss_weight * loss_node + edge_loss_weight * loss_edge
        
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss


def test_combined(model, criterion, loader,
                  threshold_structural=0.5,
                  threshold_coplanarity=0.5):
    model.eval()
    
    train_accs_struct, test_accs_struct = [], []
    train_accs_coplanarity, test_accs_coplanarity = [], []
    
    with torch.no_grad():
        for sub_data in loader:
            node_out, edge_out = model(sub_data.x, sub_data.edge_index)
            
            # Structural/Textural accuracry (add other metrics)
            preds_struct = (node_out >= threshold_structural).float()
            target_struct = sub_data.y.unsqueeze(1).float()
            for mask, acc_list in [(sub_data.train_mask, train_accs_struct), (sub_data.test_mask,  test_accs_struct)]:
                if mask.sum() > 0:
                    correct = (preds_struct[mask] == target_struct[mask]).float().sum()
                    acc_list.append((correct / mask.sum()).item())
            
            # Coplanaity accuracy (add other metrics)
            preds_coplan = (edge_out >= threshold_coplanarity).float().view(-1)
            target_coplan = sub_data.edge_labels.float().view(-1)
            
            for mask, acc_list in [(sub_data.train_edge_mask, train_accs_coplanarity), (sub_data.test_edge_mask,  test_accs_coplanarity)]:
                if mask.sum() > 0:
                    correct = (preds_coplan[mask] == target_coplan[mask]).float().sum()
                    acc_list.append((correct / mask.sum()).item())
    

    avg_train_acc_struct   = sum(train_accs_struct)   / len(train_accs_struct)
    avg_test_acc_struct    = sum(test_accs_struct)    / len(test_accs_struct)
    avg_train_acc_coplan   = sum(train_accs_coplanarity) / len(train_accs_coplanarity)
    avg_test_acc_coplan    = sum(test_accs_coplanarity)  / len(test_accs_coplanarity)
    
    return avg_train_acc_struct, avg_test_acc_struct, avg_train_acc_coplan, avg_test_acc_coplan


def run_inference(model, data_loader, model_path='model.pth', 
                  device='cpu', threshold_structural=0.5, threshold_coplanarity=0.5):
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_node_preds = []
    all_edge_preds = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            node_pred, edge_pred = model(data.x, data.edge_index)

            # Apply thresholds if you want binary predictions
            node_labels = (node_pred >= threshold_structural).float()
            edge_labels = (edge_pred >= threshold_coplanarity).float()

            all_node_preds.append(node_labels.cpu())
            all_edge_preds.append(edge_labels.cpu())

    return all_node_preds, all_edge_preds


