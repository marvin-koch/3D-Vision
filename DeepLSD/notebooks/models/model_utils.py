import torch

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


def test(model, optimizer, criterion, loader, threshold=0.5):
    model.eval()
    train_accs, val_accs, test_accs = [], [], []
    with torch.no_grad():
        for sub_data in loader:
            out = model(sub_data.x, sub_data.edge_index)
            
            preds = (out >= threshold).float()
            target = sub_data.y.unsqueeze(1).float()

            tuple_mask_accuracylist =  zip([sub_data.train_mask, sub_data.val_mask, sub_data.test_mask], [train_accs, val_accs, test_accs])
            for mask, acc_list in tuple_mask_accuracylist:
                correct = (preds[mask] == target[mask]).float().sum()
                accuracy = correct / mask.sum().float()
                acc_list.append(accuracy.item())
    avg_train_acc = sum(train_accs) / len(train_accs)
    avg_val_acc   = sum(val_accs) / len(val_accs)
    avg_test_acc  = sum(test_accs) / len(test_accs)
    return avg_train_acc, avg_val_acc, avg_test_acc


def train_combined(model, optimizer, criterion, loader, node_loss_weight=1.0, edge_loss_weight=1.0):
    model.train()
    total_loss = 0
    for sub_data in loader:
        optimizer.zero_grad()
        
        # Forward pass: get both node-level and edge-level predictions
        node_out, edge_out = model(sub_data.x, sub_data.edge_index)
        
        # --- Node-level Loss Computation ---
        node_target = sub_data.y.unsqueeze(1).float()
        loss_node = criterion(node_out[sub_data.train_mask], node_target[sub_data.train_mask])
        
        # --- Edge-level Loss Computation ---
        edge_target = sub_data.edge_labels.unsqueeze(1).float()
        
        # Here we compute the edge-level loss for every edge
        loss_edge = criterion(edge_out, edge_target)
        
        # --- Combine Losses ---
        loss = node_loss_weight * loss_node + edge_loss_weight * loss_edge
        
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss
