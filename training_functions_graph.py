import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score

def l2_loss(pred,y):
    dist = torch.sqrt(torch.sum((pred-y)**2,dim=-1))
    y_norm = torch.sqrt(torch.sum(y**2,dim=-1))
    l2 = torch.mean(dist/y_norm,dim=0)
    return l2

def r2loss(pred, y):
    SS_res = torch.sum((pred-y)**2,dim=1)
    y_mean = torch.mean(y,dim=1,keepdims=True)
    SS_tot = torch.sum((y-y_mean)**2,dim=1)
    r2 = 1 - SS_res/SS_tot
    return torch.mean(r2,dim=0)

def train(model,neighbor_centers_list,finder,optimizer,scheduler,train_loader,device,u_normalizer,config):
    if config.scope == 'global':
        train_mse, train_l2, train_r2 = train_global(model,optimizer,scheduler,train_loader,device,u_normalizer,config)
    elif config.scope == 'local':
        train_mse, train_l2, train_r2 = train_split(model,neighbor_centers_list,finder,optimizer,scheduler,train_loader,device,u_normalizer,config)
    return train_mse, train_l2, train_r2

def test(model,neighbor_centers_list,finder,test_loader,device,u_normalizer,config):
    if config.scope == 'global':
        test_mse, test_l2, test_r2 = test_global(model,test_loader,device,u_normalizer,config)
    elif config.scope == 'local':
        test_mse, test_l2, test_r2 = test_split(model,neighbor_centers_list,finder,test_loader,device,u_normalizer,config)
    return test_mse, test_l2, test_r2

def train_global(model,optimizer,scheduler,train_loader,device,u_normalizer,config):
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0
    num_sample = 0
    for i_batch, data_batch in enumerate(train_loader):
        real_num_batch = data_batch.batch[-1].item()+1
        x, edge_index, edge_attr,true_y = data_batch.x, data_batch.edge_index, data_batch.edge_attr,data_batch.y
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        true_y = true_y.to(device)
        optimizer.zero_grad()
        pred_y = model(x, edge_index, edge_attr)
        loss = torch.norm(pred_y.view(-1) - true_y.view(-1),1)
        loss.backward()
        optimizer.step()
        train_mse += F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1)).item()
        train_l2 += l2_loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        train_r2 += r2loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        num_sample += 1
    scheduler.step()
    train_mse /= num_sample
    train_l2 /= num_sample
    train_r2 /= num_sample
    return train_mse, train_l2, train_r2

def train_split(model,neighbor_centers_list,finder,optimizer,scheduler,train_loader,device,u_normalizer,config):
    model.train()
    train_mse = 0.0
    train_l2 = 0.0
    train_r2 = 0.0
    num_sample = 0

    local_mse = 0.0
    local_l2 = 0.0
    local_r2 = 0.0
    num_sample_local = 0
    for i_batch, data_batch in enumerate(train_loader):
        x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        true_y = data_batch.y.to(device)
        pred_y = true_y.new_empty((true_y.shape))
        pred_y.fill_(0)
        real_num_batch = data_batch.batch[-1].item()+1
        acc = torch.arange(0,real_num_batch*config.num_node, config.num_node,device=device)[:,None]
        for i_subgraph_batch, neighbor_centers in enumerate(neighbor_centers_list):
            real_subgraph_batch = neighbor_centers.shape[0]
            node_idx_subgraphs, edge_index_subgraphs, remap_centers,edge_enum_subgraphs = finder.find_batch_big_graph_neighbors(neighbor_centers, config.num_hops,real_num_batch)
            subgraph_node = (node_idx_subgraphs % config.num_node)+(node_idx_subgraphs // (config.num_subgraph_batch*config.num_node))*config.num_node
            x_subgraphs = x[subgraph_node]
            subgraph_edge_enum = (edge_enum_subgraphs % config.num_edge)+(edge_enum_subgraphs // (config.num_subgraph_batch*config.num_edge))*config.num_edge
            edge_attr_subgraphs = edge_attr[subgraph_edge_enum]

            x_subgraphs = x_subgraphs.to(device)
            edge_index_subgraphs = edge_index_subgraphs.to(device)
            remap_centers = remap_centers.to(device)
            edge_attr_subgraphs = edge_attr_subgraphs.to(device)
            neighbor_centers = neighbor_centers.to(device)

            optimizer.zero_grad()
            out = model(x_subgraphs, edge_index_subgraphs, edge_attr_subgraphs)

            pred_y_centers = out[remap_centers]
            batch_neighbor_centers = torch.tile(neighbor_centers,(real_num_batch,1))+acc
            batch_neighbor_centers = batch_neighbor_centers.view(-1,1)
            true_y_centers = true_y[batch_neighbor_centers]

            loss =  (out.view(-1)-true_y[subgraph_node].view(-1))**2
            loss[remap_centers] = loss[remap_centers]*2
            loss = torch.sum(loss)
            #loss = torch.norm(out.view(-1)-true_y[subgraph_node].view(-1),1 )
            #mse = F.mse_loss(out.view(-1, 1), true_y[subgraph_node].view(-1,1))
            #mse = F.mse_loss(out.view(real_num_batch, -1), true_y[subgraph_node].view(real_num_batch,-1),reduction='mean')
            #loss = torch.norm(pred_y_centers.view(-1) - true_y_centers.view(-1),1)

            loss.backward()
            optimizer.step()
            pred_y[batch_neighbor_centers] = pred_y_centers

            local_mse += F.mse_loss(out.view(-1, 1), true_y[subgraph_node].view(-1,1)).item()
            local_l2 += l2_loss(u_normalizer.decode(out.view(real_num_batch,-1)), u_normalizer.decode(true_y[subgraph_node].view(real_num_batch, -1))).item()
            local_r2 += r2loss(u_normalizer.decode(out.view(real_num_batch,-1)), u_normalizer.decode(true_y[subgraph_node].view(real_num_batch, -1))).item()
            num_sample_local += 1

        scheduler.step()
        train_mse += F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1)).item()
        train_l2 += l2_loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        train_r2 += r2loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        num_sample += 1
    train_mse /= num_sample
    train_l2 /= num_sample
    train_r2 /= num_sample

    local_mse /= num_sample_local
    local_l2 /= num_sample_local
    local_r2 /= num_sample_local
    print('local train_mse: {:.8f},train_l2: {:.8f}, train_r2: {:.8f}'.format(local_mse,local_l2, local_r2))
    return train_mse, train_l2, train_r2

def test_global(model,test_loader,device,u_normalizer,config):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0  
    num_sample = 0   
    with torch.no_grad():
        for i_batch, data_batch in enumerate(test_loader):
            real_num_batch = data_batch.batch[-1].item()+1
            x, edge_index, edge_attr,true_y = data_batch.x, data_batch.edge_index, data_batch.edge_attr,data_batch.y
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            true_y = true_y.to(device)
            pred_y = model(x, edge_index, edge_attr)
            test_mse += F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1)).item()
            test_l2 += l2_loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
            test_r2 += r2loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
            num_sample += 1
    test_mse /= num_sample
    test_l2 /= num_sample
    test_r2 /= num_sample
    return test_mse, test_l2, test_r2

def test_split(model,neighbor_centers_list,finder,test_loader,device,u_normalizer,config):
    model.eval()              
    test_mse = 0.0
    test_l2 = 0.0
    test_r2 = 0.0  
    num_sample = 0   
    with torch.no_grad():
        for i_batch, data_batch in enumerate(test_loader):
            x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
        true_y = data_batch.y.to(device)
        pred_y = true_y.new_empty((true_y.shape))
        pred_y.fill_(0)
        real_num_batch = data_batch.batch[-1].item()+1
        acc = torch.arange(0,real_num_batch*config.num_node, config.num_node,device=device)[:,None]
        for i_subgraph_batch, neighbor_centers in enumerate(neighbor_centers_list):
            node_idx_subgraphs, edge_index_subgraphs, remap_centers,edge_enum_subgraphs = finder.find_batch_big_graph_neighbors(neighbor_centers, config.num_hops,real_num_batch)
            subgraph_node = (node_idx_subgraphs % config.num_node)+(node_idx_subgraphs // (config.num_subgraph_batch*config.num_node))*config.num_node
            x_subgraphs = x[subgraph_node]
            subgraph_edge_enum = (edge_enum_subgraphs % config.num_edge)+(edge_enum_subgraphs // (config.num_subgraph_batch*config.num_edge))*config.num_edge
            edge_attr_subgraphs = edge_attr[subgraph_edge_enum]

            x_subgraphs = x_subgraphs.to(device)
            edge_index_subgraphs = edge_index_subgraphs.to(device)
            remap_centers = remap_centers.to(device)
            edge_attr_subgraphs = edge_attr_subgraphs.to(device)
            neighbor_centers = neighbor_centers.to(device)
            out = model(x_subgraphs, edge_index_subgraphs, edge_attr_subgraphs)

            pred_y_centers = out[remap_centers]
            batch_neighbor_centers = torch.tile(neighbor_centers,(real_num_batch,1))+acc
            batch_neighbor_centers = batch_neighbor_centers.view(-1,1)

            pred_y[batch_neighbor_centers] = pred_y_centers
        test_mse += F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1)).item()
        test_l2 += l2_loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        test_r2 += r2loss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1))).item()
        num_sample += 1
    test_mse /= num_sample
    test_l2 /= num_sample
    test_r2 /= num_sample
    return test_mse, test_l2, test_r2