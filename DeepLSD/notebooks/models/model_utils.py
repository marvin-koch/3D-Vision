import torch

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
        print(f"Loss for mini-batch {batch_n}: {loss.item()}")
        batch_n += 1
        total_loss += loss.item()

    return total_loss / len(loader)


def test_inductive(model, loader, device='cpu', threshold_structural=0.5, threshold_coplanarity=0.5):
    model.eval()
    correct_nodes = total_nodes = 0
    correct_edges = total_edges = 0
    batch_n = 0

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            node_pred, edge_pred = model(data.x, data.edge_index)

            # node accuracy
            pred_n = (node_pred >= threshold_structural).float()
            correct_nodes += (pred_n == data.y).sum().item()
            total_nodes   += data.y.numel()

            # edge accuracy
            pred_e = (edge_pred >= threshold_coplanarity).float()
            correct_edges += (pred_e == data.edge_labels).sum().item()
            total_edges   += data.edge_labels.numel()
            print(f"Correct nodes for mini-batch {batch_n}: {correct_nodes}")
            print(f"Correct edges for mini-batch {batch_n}: {correct_edges}")
            batch_n += 1

    node_acc = correct_nodes / total_nodes
    edge_acc = correct_edges / total_edges
    return node_acc, edge_acc



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

