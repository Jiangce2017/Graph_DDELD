import torch
import numpy as np

import torch.nn.functional as F

from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from utilities import *
from nn_conv import NNConv_old

from timeit import default_timer

from find_neighbors_model import Neihbor_finder
from visualize import plot_biggraph,plot_subgrpahs

torch.manual_seed(1234)
#### load data

TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

r = 4
s = int(((241 - 1)/r) + 1)
n = s**2
m = 100
k = 1

radius_train = 0.1
radius_test = 0.1

print('resolution', s)


ntrain = 100
ntest = 40

batch_size = 1
batch_size2 = 2
width = 64
ker_width = 1024
depth = 6
edge_features = 6
node_features = 6

epochs = 200
learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.8

path = 'UAI1_r'+str(s)+'_n'+ str(ntrain)
path_model = 'model/'+path+''
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path+''
path_train_err = 'results/'+path+'train'
path_test_err16 = 'results/'+path+'test16'
path_test_err31 = 'results/'+path+'test31'
path_test_err61 = 'results/'+path+'test61'
path_image_train = 'image/'+path+'train'
path_image_test16 = 'image/'+path+'test16'
path_image_test31 = 'image/'+path+'test31'
path_image_test61 = 'image/'+path+'test61'

t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u64 = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::4,::4].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::4,::4].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::4,::4].reshape(ntest,-1)


a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)


test_a = test_a.reshape(ntest,61,61)
test_a_smooth = test_a_smooth.reshape(ntest,61,61)
test_a_gradx = test_a_gradx.reshape(ntest,61,61)
test_a_grady = test_a_grady.reshape(ntest,61,61)
test_u = test_u.reshape(ntest,61,61)

test_a16 =test_a[:ntest,::4,::4].reshape(ntest,-1)
test_a_smooth16 = test_a_smooth[:ntest,::4,::4].reshape(ntest,-1)
test_a_gradx16 = test_a_gradx[:ntest,::4,::4].reshape(ntest,-1)
test_a_grady16 = test_a_grady[:ntest,::4,::4].reshape(ntest,-1)
test_u16 = test_u[:ntest,::4,::4].reshape(ntest,-1)

u_normalizer = GaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)


meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
edge_index = meshgenerator.ball_connectivity(radius_train)
grid = meshgenerator.get_grid()
data_train = []
for j in range(ntrain):
    edge_attr = meshgenerator.attributes(theta=train_a[j,:])
    data_train.append(Data(x=torch.cat([grid, train_a[j,:].reshape(-1, 1),
                                        train_a_smooth[j,:].reshape(-1, 1), train_a_gradx[j,:].reshape(-1, 1), train_a_grady[j,:].reshape(-1, 1)
                                        ], dim=1),
                           y=train_u[j,:], coeff=train_a[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                           ))
print('train grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[16,16])
edge_index = meshgenerator.ball_connectivity(radius_test)
grid = meshgenerator.get_grid()
# meshgenerator.get_boundary()
# edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
data_test16 = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a16[j,:])
    # edge_attr_boundary = meshgenerator.attributes_boundary(theta=test_a[j, :])
    data_test16.append(Data(x=torch.cat([grid, test_a16[j,:].reshape(-1, 1),
                                       test_a_smooth16[j,:].reshape(-1, 1), test_a_gradx16[j,:].reshape(-1, 1), test_a_grady16[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=test_u16[j, :], coeff=test_a16[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                           # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
                          ))

print('16 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader16 = DataLoader(data_test16, batch_size=batch_size2, shuffle=False)

###### start testing #########
device = torch.device('cuda:0')

data_iterator = iter(test_loader16)
data_batch = next(data_iterator)
data_batch = data_batch.to(device)
batch, x, edge_index, edge_attr = data_batch.batch, data_batch.x, data_batch.edge_index, data_batch.edge_attr

num_batch = batch[-1].item()+1
num_subgraph_batch = 4
num_node = x.shape[0]//num_batch
num_edge = edge_index.shape[1]//num_batch
print("single graph num_node: {}".format(num_node))
print("single graph num_edge: {}".format(num_edge))
print("batch number: {}".format(num_batch))
edge_index_single = edge_index[:,edge_index[0,:]<num_node]
finder = Neihbor_finder(num_node,num_edge, edge_index_single,num_subgraph_batch, num_batch)

full_node_idx = torch.randperm(num_node,device=device) 
neighbor_centers_list = torch.split(full_node_idx,num_subgraph_batch,dim=0)

neighbor_centers = neighbor_centers_list[0]
num_hops = 2

if num_batch == 1:
    ### test single big graph
    node_idx_subgraphs, edge_index_subgraphs, remap_centers,edge_enum_subgraphs = finder.find_single_big_graph_neighbors(neighbor_centers, num_hops)
    ###  visualize subgraphs
    figure_path_biggraph = './image/single_big_graph.jpg'
    # plot the big graph
    points = x[batch==0,:2]
    points = points.cpu().detach().numpy()
    edge_index = edge_index.cpu().detach().numpy()
    node_idx_subgraphs = node_idx_subgraphs.cpu().detach().numpy()
    edge_index_subgraphs = edge_index_subgraphs.cpu().detach().numpy()
    remap_centers = remap_centers.cpu().detach().numpy()
    plot_biggraph(points, edge_index, figure_path_biggraph)

    ## plot big graph with subgraphs
    figure_path_subgraphs = './image/single_subgraphs.jpg'
    sub_graph_node = node_idx_subgraphs % num_node
    sub_points = points[sub_graph_node,:]
    plot_subgrpahs(remap_centers, points, edge_index,sub_points,edge_index_subgraphs,figure_path_subgraphs)
elif num_batch > 1:
    node_idx_subgraphs, edge_index_subgraphs, remap_centers,edge_enum_subgraphs = finder.find_batch_big_graph_neighbors(neighbor_centers, num_hops,num_batch)
    figure_path_biggraph = './image/batch_big_graph.jpg'
    points = x.new_empty((x.shape[0],2))
    for i_batch in range(num_batch):
        points[batch==i_batch,:] = x[batch==i_batch,:2] 
        points[batch==i_batch,0] += i_batch*1.1
    points = points.cpu().detach().numpy()
    edge_index = edge_index.cpu().detach().numpy()
    node_idx_subgraphs = node_idx_subgraphs.cpu().detach().numpy()
    edge_index_subgraphs = edge_index_subgraphs.cpu().detach().numpy()
    remap_centers = remap_centers.cpu().detach().numpy()
    plot_biggraph(points, edge_index, figure_path_biggraph)

    ## plot big graph with subgraphs
    figure_path_subgraphs = './image/batch_subgraphs.jpg'
    ## mapping node from subgraphs to big graphs
    subgraph_node = (node_idx_subgraphs % num_node)+(node_idx_subgraphs // (num_subgraph_batch*num_node))*num_node
    sub_points = points[subgraph_node,:]

    subgraph_edge_enum = (edge_enum_subgraphs % num_edge)+(edge_enum_subgraphs // (num_subgraph_batch*num_edge))*num_edge
    edge_attr_subgraphs = edge_attr[subgraph_edge_enum]
    print("edge_attr_subgraphs shape: {}".format(edge_attr_subgraphs.shape))
    print("edge_index_subgraphs shape: {}".format(edge_index_subgraphs.shape))
    plot_subgrpahs(remap_centers, points, edge_index,sub_points,edge_index_subgraphs,figure_path_subgraphs)