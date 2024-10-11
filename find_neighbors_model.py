import torch

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Neihbor_finder(torch.nn.Module):
    def __init__(self, num_node,num_edge, edge_index,num_subgraph_batch, num_batch, flow='source_to_target'):
        '''
        the num_node, num_edge_edge_index are all respect to a single graph, not batch of graphs
        '''
        super(torch.nn.Module, self).__init__()
        self.num_node = num_node
        self.directed = False
        assert flow in ['source_to_target', 'target_to_source']
        if flow == 'source_to_target': ## ?? maybe swap
            self.row, self.col = edge_index
        else:
            self.col, self.row = edge_index
        self.num_edge = num_edge
        self.unique_col, self.inv_idx = torch.unique(self.col, return_inverse=True)
        self.num_subgraph_batch = num_subgraph_batch
        self.node_idx = torch.arange(num_node,device=self.row.device)
        self.edge_enum = torch.arange(self.num_edge,device=self.row.device)
        self.num_batch = num_batch
        self.subgraph_acc = torch.arange(0,num_subgraph_batch*num_node, num_node,device=self.row.device)[:,None]
        self.subgraph_acc_edge = torch.arange(0,num_subgraph_batch*num_edge, num_edge,device=self.row.device)[:,None]

        self.node_idx_large, self.edge_index_large, self.edge_enum_large = self.broadcast_graphs(self.node_idx,self.edge_enum,self.row,self.col,self.num_subgraph_batch,self.subgraph_acc,self.subgraph_acc_edge)
        
        self.biggraph_acc = torch.arange(0,num_batch*num_subgraph_batch*num_node, num_node*num_subgraph_batch,device=self.row.device)[:,None]
        self.biggraph_acc_edge = torch.arange(0,num_batch*num_subgraph_batch*num_edge, num_edge*num_subgraph_batch,device=self.row.device)[:,None]

    def broadcast_graphs(self,node_idx,edge_enum, row,col, num_graph, acc,acc_edge):
        row_large = torch.tile(row,(num_graph,1))
        col_large = torch.tile(col,(num_graph,1))
        row_large += acc
        col_large += acc
        edge_index_large = torch.cat((row_large.view(1,-1),col_large.view(1,-1)),dim=0)
        node_idx_large = torch.tile(node_idx,(num_graph,1))
        node_idx_large += acc
        node_idx_large = node_idx_large.view(-1)
        edge_enum_large = torch.tile(edge_enum,(num_graph,1))
        edge_enum_large += acc_edge
        edge_enum_large = edge_enum_large.view(-1)
        return node_idx_large, edge_index_large,edge_enum_large
    
    
    def find_single_big_graph_neighbors(self, neighbor_centers, num_hops, relabel_nodes=True):
        '''
        neighbor_centers is a 1D tensor
        output is node_mask (num_subgraph_batch x num_node ) and edge_mask (num_subgraph_batch x num_edge)
        '''
        if self.num_subgraph_batch == neighbor_centers.shape[0]:
            num_subgraph_batch = self.num_subgraph_batch
            edge_index_large = self.edge_index_large
            node_idx_large = self.node_idx_large
            edge_enum_large = self.edge_enum_large
            subgraph_acc = self.subgraph_acc
        else:
            num_subgraph_batch = neighbor_centers.shape[0]
            subgraph_acc = torch.arange(0,num_subgraph_batch*self.num_node, self.num_node,device=self.row.device)[:,None]
            subgraph_acc_edge = torch.arange(0,num_subgraph_batch*self.num_edge, self.num_edge,device=self.row.device)[:,None]
            node_idx_large, edge_index_large,edge_enum_large = self.broadcast_graphs(self.node_idx,self.edge_enum, self.row,self.col,num_subgraph_batch,subgraph_acc,subgraph_acc_edge)

        node_mask = self.row.new_empty((num_subgraph_batch,self.num_node),dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[torch.arange(num_subgraph_batch,device=self.row.device),neighbor_centers] = True
        edge_mask = self.row.new_empty((num_subgraph_batch, self.row.size(0)), dtype=torch.bool)
        zero_sum = self.row.new_empty((num_subgraph_batch,self.unique_col.shape[0]), dtype=torch.bool)
        zero_sum.fill_(False)
        for _ in range(num_hops):
            torch.index_select(node_mask, 1, self.row, out=edge_mask)
            unique_edge_mask = zero_sum.index_add_(1, self.inv_idx, edge_mask)
            node_mask[:,self.unique_col] += unique_edge_mask

        ## minibatch
        if not self.directed:
            edge_mask = node_mask[:,self.row] & node_mask[:,self.col]    
        edge_index_subgraphs = edge_index_large[:,edge_mask.view(-1)]
        node_idx_subgraphs = node_idx_large[node_mask.view(-1)] ## node_idx_large could be replaced by node_mask

        new_node_idx = self.row.new_full((self.num_node*num_subgraph_batch, ), -1)
        new_node_idx[node_idx_subgraphs] = torch.arange(node_idx_subgraphs.size(0), device=self.row.device)
        ## remapping neighbor centers
        remap_centers = new_node_idx[neighbor_centers+subgraph_acc[:,0]]

        if relabel_nodes:
            edge_index_subgraphs = new_node_idx[edge_index_subgraphs]

        edge_enum_subgraphs = edge_enum_large[edge_mask.view(-1)]

        return node_idx_subgraphs, edge_index_subgraphs, remap_centers, edge_enum_subgraphs

    def find_batch_big_graph_neighbors(self, neighbor_centers, num_hops,real_num_batch):
        num_batch = real_num_batch
        num_subgraph_batch = neighbor_centers.shape[0]
        if self.num_batch == real_num_batch and self.num_subgraph_batch == neighbor_centers.shape[0]:
            acc = self.biggraph_acc
            acc_edge = self.biggraph_acc_edge
        else:
            acc = torch.arange(0,num_batch*num_subgraph_batch*self.num_node, self.num_node*num_subgraph_batch,device=self.row.device)[:,None]
            acc_edge = torch.arange(0,num_batch*num_subgraph_batch*self.num_edge, self.num_edge*num_subgraph_batch,device=self.row.device)[:,None]
        node_idx_subgraphs, edge_index_subgraphs,remap_centers_single, edge_enum_subgraphs = self.find_single_big_graph_neighbors(neighbor_centers, num_hops,relabel_nodes=False)
        node_idx_large, edge_index_large, edge_enum_large = self.broadcast_graphs(node_idx_subgraphs,edge_enum_subgraphs,edge_index_subgraphs[0,:],edge_index_subgraphs[1,:], num_batch,acc,acc_edge)
        
        ## relabel nodes
        new_node_idx = self.row.new_full((self.num_node*num_subgraph_batch*num_batch, ), -1)
        new_node_idx[node_idx_large] = torch.arange(node_idx_large.size(0), device=self.row.device)
        edge_index_large = new_node_idx[edge_index_large]
       
        ## remapping neighbor centers
        num_subgraph_node = node_idx_subgraphs.shape[0] # number of subgraph nodes in each single big graph
        center_acc = torch.arange(0,num_batch*num_subgraph_node, num_subgraph_node,device=self.row.device)[:,None]
        remap_centers_batch = torch.tile(remap_centers_single,(num_batch,1))
        remap_centers_batch += center_acc
        remap_centers_batch = remap_centers_batch.view(-1)

        return node_idx_large, edge_index_large, remap_centers_batch,edge_enum_large
    
