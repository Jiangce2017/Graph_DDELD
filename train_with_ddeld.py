import torch
import numpy as np

import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
from nn_conv import NNConv_old

from timeit import default_timer
from find_neighbors_model import Neihbor_finder

torch.manual_seed(1234)

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, x, edge_index, edge_attr):
        #x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


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

batch_size = 8
batch_size2 = 2
width = 16 #64
ker_width = 128 #1024
depth = 4 #6
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

# reader.load_file(TEST_PATH)
# test_a = reader.read_field('coeff')[:ntest,::4,::4].reshape(ntest,-1)
# test_a_smooth = reader.read_field('Kcoeff')[:ntest,::4,::4].reshape(ntest,-1)
# test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::4,::4].reshape(ntest,-1)
# test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::4,::4].reshape(ntest,-1)
# test_u = reader.read_field('sol')[:ntest,::4,::4].reshape(ntest,-1)


a_normalizer = GaussianNormalizer(train_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)


train_a = a_normalizer.encode(train_a)
train_a_smooth = as_normalizer.encode(train_a_smooth)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
train_a_grady = agy_normalizer.encode(train_a_grady)


train_a = train_a.reshape(ntrain,61,61)
train_a_smooth = train_a_smooth.reshape(ntrain,61,61)
train_a_gradx = train_a_gradx.reshape(ntrain,61,61)
train_a_grady = train_a_grady.reshape(ntrain,61,61)
train_u = train_u.reshape(ntrain,61,61)


# train_a16 =train_a[:ntrain,::4,::4].reshape(ntrain,-1)
# train_a_smooth16 = train_a_smooth[:ntrain,::4,::4].reshape(ntrain,-1)
# train_a_gradx16 = train_a_gradx[:ntrain,::4,::4].reshape(ntrain,-1)
# train_a_grady16 = train_a_grady[:ntrain,::4,::4].reshape(ntrain,-1)
# train_u16 = train_u[:ntrain,::4,::4].reshape(ntrain,-1)

train_a31 =train_a[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_smooth31 = train_a_smooth[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_gradx31 = train_a_gradx[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_grady31 = train_a_grady[:ntrain,::2,::2].reshape(ntrain,-1)
train_u31 = train_u[:ntrain,::2,::2].reshape(ntrain,-1)

# test_a = a_normalizer.encode(test_a)
# test_a_smooth = as_normalizer.encode(test_a_smooth)
# test_a_gradx = agx_normalizer.encode(test_a_gradx)
# test_a_grady = agy_normalizer.encode(test_a_grady)


# test_a = test_a.reshape(ntest,61,61)
# test_a_smooth = test_a_smooth.reshape(ntest,61,61)
# test_a_gradx = test_a_gradx.reshape(ntest,61,61)
# test_a_grady = test_a_grady.reshape(ntest,61,61)
# test_u = test_u.reshape(ntest,61,61)

# test_a16 =test_a[:ntest,::4,::4].reshape(ntest,-1)
# test_a_smooth16 = test_a_smooth[:ntest,::4,::4].reshape(ntest,-1)
# test_a_gradx16 = test_a_gradx[:ntest,::4,::4].reshape(ntest,-1)
# test_a_grady16 = test_a_grady[:ntest,::4,::4].reshape(ntest,-1)
# test_u16 = test_u[:ntest,::4,::4].reshape(ntest,-1)

# test_a31 =test_a[:ntest,::2,::2].reshape(ntest,-1)
# test_a_smooth31 = test_a_smooth[:ntest,::2,::2].reshape(ntest,-1)
# test_a_gradx31 = test_a_gradx[:ntest,::2,::2].reshape(ntest,-1)
# test_a_grady31 = test_a_grady[:ntest,::2,::2].reshape(ntest,-1)
# test_u31 = test_u[:ntest,::2,::2].reshape(ntest,-1)

# test_a =test_a.reshape(ntest,-1)
# test_a_smooth = test_a_smooth.reshape(ntest,-1)
# test_a_gradx = test_a_gradx.reshape(ntest,-1)
# test_a_grady = test_a_grady.reshape(ntest,-1)
# test_u = test_u.reshape(ntest,-1)


u_normalizer = GaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)



# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
# edge_index = meshgenerator.ball_connectivity(radius_train)
# grid = meshgenerator.get_grid() 
# #meshgenerator.get_boundary()
# #edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
# data_train = []
# for j in range(ntrain):
#     edge_attr = meshgenerator.attributes(theta=train_a[j,:])
#     # edge_attr_boundary = meshgenerator.attributes_boundary(theta=train_u[j,:])
#     data_train.append(Data(x=torch.cat([grid, train_a[j,:].reshape(-1, 1),
#                                         train_a_smooth[j,:].reshape(-1, 1), train_a_gradx[j,:].reshape(-1, 1), train_a_grady[j,:].reshape(-1, 1)
#                                         ], dim=1),
#                            y=train_u[j,:], coeff=train_a[j,:],
#                            edge_index=edge_index, edge_attr=edge_attr,
#                            # edge_index_boundary=edge_index_boundary, edge_attr_boundary= edge_attr_boundary
#                            ))

# print('train grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)


meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[31,31])
edge_index = meshgenerator.ball_connectivity(radius_train)
grid = meshgenerator.get_grid()
# meshgenerator.get_boundary()
# edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
data_train31 = []
for j in range(ntrain):
    edge_attr = meshgenerator.attributes(theta=train_a31[j,:])
    # edge_attr_boundary = meshgenerator.attributes_boundary(theta=train_a[j, :])
    data_train31.append(Data(x=torch.cat([grid, train_a31[j,:].reshape(-1, 1),
                                       train_a_smooth31[j,:].reshape(-1, 1), train_a_gradx31[j,:].reshape(-1, 1), train_a_grady31[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=train_u31[j, :], coeff=train_a31[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                           # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
                          ))

print('31 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)



# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[16,16])
# edge_index = meshgenerator.ball_connectivity(radius_train)
# grid = meshgenerator.get_grid()
# # meshgenerator.get_boundary()
# # edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
# data_train16 = []
# for j in range(ntrain):
#     edge_attr = meshgenerator.attributes(theta=train_a16[j,:])
#     # edge_attr_boundary = meshgenerator.attributes_boundary(theta=train_a[j, :])
#     data_train16.append(Data(x=torch.cat([grid, train_a16[j,:].reshape(-1, 1),
#                                        train_a_smooth16[j,:].reshape(-1, 1), train_a_gradx16[j,:].reshape(-1, 1), train_a_grady16[j,:].reshape(-1, 1)
#                                        ], dim=1),
#                            y=train_u16[j, :], coeff=train_a16[j,:],
#                            edge_index=edge_index, edge_attr=edge_attr,
#                            # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
#                           ))

# print('16 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# # print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)


# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[16,16])
# edge_index = meshgenerator.ball_connectivity(radius_test)
# grid = meshgenerator.get_grid()
# # meshgenerator.get_boundary()
# # edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
# data_test16 = []
# for j in range(ntest):
#     edge_attr = meshgenerator.attributes(theta=test_a16[j,:])
#     # edge_attr_boundary = meshgenerator.attributes_boundary(theta=test_a[j, :])
#     data_test16.append(Data(x=torch.cat([grid, test_a16[j,:].reshape(-1, 1),
#                                        test_a_smooth16[j,:].reshape(-1, 1), test_a_gradx16[j,:].reshape(-1, 1), test_a_grady16[j,:].reshape(-1, 1)
#                                        ], dim=1),
#                            y=test_u16[j, :], coeff=test_a16[j,:],
#                            edge_index=edge_index, edge_attr=edge_attr,
#                            # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
#                           ))

# print('16 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# # print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)

# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[31,31])
# edge_index = meshgenerator.ball_connectivity(radius_test)
# grid = meshgenerator.get_grid()
# # meshgenerator.get_boundary()
# # edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
# data_test31 = []
# for j in range(ntest):
#     edge_attr = meshgenerator.attributes(theta=test_a31[j,:])
#     # edge_attr_boundary = meshgenerator.attributes_boundary(theta=test_a[j, :])
#     data_test31.append(Data(x=torch.cat([grid, test_a31[j,:].reshape(-1, 1),
#                                        test_a_smooth31[j,:].reshape(-1, 1), test_a_gradx31[j,:].reshape(-1, 1), test_a_grady31[j,:].reshape(-1, 1)
#                                        ], dim=1),
#                            y=test_u31[j, :], coeff=test_a31[j,:],
#                            edge_index=edge_index, edge_attr=edge_attr,
#                            # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
#                           ))

# print('31 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# # print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)

# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[61,61])
# edge_index = meshgenerator.ball_connectivity(radius_test)
# grid = meshgenerator.get_grid()
# # meshgenerator.get_boundary()
# # edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
# data_test61 = []
# for j in range(ntest):
#     edge_attr = meshgenerator.attributes(theta=test_a[j,:])
#     # edge_attr_boundary = meshgenerator.attributes_boundary(theta=test_a[j, :])
#     data_test61.append(Data(x=torch.cat([grid, test_a[j,:].reshape(-1, 1),
#                                        test_a_smooth[j,:].reshape(-1, 1), test_a_gradx[j,:].reshape(-1, 1), test_a_grady[j,:].reshape(-1, 1)
#                                        ], dim=1),
#                            y=test_u[j, :], coeff=test_a[j,:],
#                            edge_index=edge_index, edge_attr=edge_attr,
#                            # edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary
#                           ))

# print('61 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# # print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)

#train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(data_train31, batch_size=batch_size, shuffle=True)
#train_loader = DataLoader(data_train16, batch_size=batch_size, shuffle=True)
# test_loader16 = DataLoader(data_test16, batch_size=batch_size2, shuffle=False)
# test_loader31 = DataLoader(data_test31, batch_size=batch_size2, shuffle=False)
# test_loader61 = DataLoader(data_test61, batch_size=batch_size2, shuffle=False)



##################################################################################################

### training

##################################################################################################
t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda:0')

model = KernelNN(width,ker_width,depth,edge_features,node_features)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()

### Initialize Neighbor_finder
data_iterator = iter(train_loader)
data_batch = next(data_iterator)
batch, x, edge_index, edge_attr = data_batch.batch, data_batch.x, data_batch.edge_index, data_batch.edge_attr
num_batch = batch[-1].item()+1
num_subgraph_batch = 16
num_node = x.shape[0]//num_batch
num_edge = edge_index.shape[1]//num_batch
print("single graph num_node: {}".format(num_node))
print("single graph num_edge: {}".format(num_edge))
print("batch number: {}".format(num_batch))
edge_index_single = edge_index[:,edge_index[0,:]<num_node]
finder = Neihbor_finder(num_node,num_edge, edge_index_single,num_subgraph_batch, num_batch)
num_hops = 3
full_node_idx = torch.randperm(num_node) 
neighbor_centers_list = torch.split(full_node_idx,num_subgraph_batch,dim=0)

model.train()
ttrain = np.zeros((epochs, ))
ttest16 = np.zeros((epochs,))
ttest31 = np.zeros((epochs,))
ttest61 = np.zeros((epochs,))

scope = 'local'

if scope == 'local':
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0
        num_subgraph = 0
        for i_batch, data_batch in enumerate(train_loader):
            x, edge_index, edge_attr = data_batch.x, data_batch.edge_index, data_batch.edge_attr
            true_y = data_batch.y.to(device)
            pred_y = true_y.new_empty((true_y.shape))
            pred_y.fill_(0)
            real_num_batch = data_batch.batch[-1].item()+1
            for i_subgraph_batch, neighbor_centers in enumerate(neighbor_centers_list):
                node_idx_subgraphs, edge_index_subgraphs, remap_centers,edge_enum_subgraphs = finder.find_batch_big_graph_neighbors(neighbor_centers, num_hops,real_num_batch)
                subgraph_node = (node_idx_subgraphs % num_node)+(node_idx_subgraphs // (num_subgraph_batch*num_node))*num_node
                x_subgraphs = x[subgraph_node]
                subgraph_edge_enum = (edge_enum_subgraphs % num_edge)+(edge_enum_subgraphs // (num_subgraph_batch*num_edge))*num_edge
                edge_attr_subgraphs = edge_attr[subgraph_edge_enum]

                x_subgraphs = x_subgraphs.to(device)
                edge_index_subgraphs = edge_index_subgraphs.to(device)
                remap_centers = remap_centers.to(device)
                edge_attr_subgraphs = edge_attr_subgraphs.to(device)
                optimizer.zero_grad()
                out = model(x_subgraphs, edge_index_subgraphs, edge_attr_subgraphs)

                pred_y_centers = out[remap_centers]
                batch_neighbor_centers = torch.tile(neighbor_centers,(1,real_num_batch)).view(-1,1)
                true_y_centers = true_y[batch_neighbor_centers]
                mse = F.mse_loss(pred_y_centers.view(-1, 1), true_y_centers.view(-1,1))
                # mse.backward() 
                loss = torch.norm(pred_y_centers.view(-1) - true_y_centers.view(-1),1)
                loss.backward()

                l2 = myloss(u_normalizer.decode(pred_y_centers.view(real_num_batch,-1)), u_normalizer.decode(true_y_centers.view(real_num_batch, -1)))
                # l2.backward()
                pred_y[batch_neighbor_centers] = pred_y_centers

                optimizer.step()
                train_mse += mse.item()
                train_l2 += l2.item()
                num_subgraph += 1
                #print("ep: {}, i_batch: {}, i_subgraph_batch: {}, mes: {}, loss: {}, l2: {}".format(ep,i_batch, i_subgraph_batch, mse.item(), loss.item(), l2.item()))
            global_mse = F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1))
            global_loss = torch.norm(pred_y.view(-1) - true_y.view(-1),1)
            global_l2 = myloss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1)))
            print("ep: {}, i_batch: {}, global_mse: {}".format(ep, i_batch, global_mse))
        scheduler.step()

elif scope == 'global':
    for ep in range(1000):
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0
        num_subgraph = 0
        for i_batch, data_batch in enumerate(train_loader):
            real_num_batch = data_batch.batch[-1].item()+1
            x, edge_index, edge_attr,true_y = data_batch.x, data_batch.edge_index, data_batch.edge_attr,data_batch.y
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            true_y = true_y.to(device)
            pred_y = model(x, edge_index, edge_attr)
            mse = F.mse_loss(pred_y.view(-1, 1), true_y.view(-1,1))
            # mse.backward() 
            loss = torch.norm(pred_y.view(-1) - true_y.view(-1),1)
            loss.backward()

            l2 = myloss(u_normalizer.decode(pred_y.view(real_num_batch,-1)), u_normalizer.decode(true_y.view(real_num_batch, -1)))
            # l2.backward()
            print("ep: {}, i_batch: {}, global_mse: {}".format(ep, i_batch, mse))
        scheduler.step()

    # t2 = default_timer()

    # model.eval()


    # ttrain[ep] = train_l2/(ntrain * k)

    #print(ep, ' time:', t2-t1, ' train_mse:', train_mse/num_subgraph)

# t1 = default_timer()
# u_normalizer.cpu()
# model = model.cpu()
# test_l2_16 = 0.0
# test_l2_31 = 0.0
# test_l2_61 = 0.0
# with torch.no_grad():
#     for data_batch in test_loader16:
#         out = model(data_batch)
#         test_l2_16 += myloss(u_normalizer.decode(out.view(batch_size2,-1)),
#                              data_batch.y.view(batch_size2, -1))
#     for data_batch in test_loader31:
#         out = model(data_batch)
#         test_l2_31 += myloss(u_normalizer.decode(out.view(batch_size2, -1)),
#                              data_batch.y.view(batch_size2, -1))
#     for data_batch in test_loader61:
#         out = model(data_batch)
#         test_l2_61 += myloss(u_normalizer.decode(out.view(batch_size2, -1)),
#                              data_batch.y.view(batch_size2, -1))

# ttest16[ep] = test_l2_16 / ntest
# ttest31[ep] = test_l2_31 / ntest
# ttest61[ep] = test_l2_61 / ntest
# t2 = default_timer()

# print(' time:', t2-t1, ' train_mse:', train_mse/len(train_loader),
#           ' test16:', test_l2_16/ntest,  ' test31:', test_l2_31/ntest,  ' test61:', test_l2_61/ntest)
# np.savetxt(path_train_err + '.txt', ttrain)
# np.savetxt(path_test_err16 + '.txt', ttest16)
# np.savetxt(path_test_err31 + '.txt', ttest31)
# np.savetxt(path_test_err61 + '.txt', ttest61)

# torch.save(model, path_model)

# ##################################################################################################

# ### Ploting

# ##################################################################################################



# resolution = s
# data = train_loader.dataset[0]
# coeff = data.coeff.numpy().reshape((resolution, resolution))
# truth = u_normalizer.decode(data.y.reshape(1,-1)).numpy().reshape((resolution, resolution))
# approx = u_normalizer.decode(model(data).reshape(1,-1)).detach().numpy().reshape((resolution, resolution))
# _min = np.min(np.min(truth))
# _max = np.max(np.max(truth))

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(truth, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Ground Truth')

# plt.subplot(1, 3, 2)
# plt.imshow(approx, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Approximation')

# plt.subplot(1, 3, 3)
# plt.imshow((approx - truth) ** 2)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Error')

# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.savefig(path_image_train + '.png')


# resolution = 16
# data = test_loader16.dataset[0]
# coeff = data.coeff.numpy().reshape((resolution, resolution))
# truth = data.y.numpy().reshape((resolution, resolution))
# approx = u_normalizer.decode(model(data).reshape(1,-1)).detach().numpy().reshape((resolution, resolution))
# _min = np.min(np.min(truth))
# _max = np.max(np.max(truth))

# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(truth, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Ground Truth')

# plt.subplot(1, 3, 2)
# plt.imshow(approx, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Approximation')

# plt.subplot(1, 3, 3)
# plt.imshow((approx - truth) ** 2)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Error')

# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.savefig(path_image_test16 + '.png')

# resolution = 31
# data = test_loader31.dataset[0]
# coeff = data.coeff.numpy().reshape((resolution, resolution))
# truth = data.y.numpy().reshape((resolution, resolution))
# approx = u_normalizer.decode(model(data).reshape(1,-1)).detach().numpy().reshape((resolution, resolution))
# _min = np.min(np.min(truth))
# _max = np.max(np.max(truth))

# # plt.figure()
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(truth, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Ground Truth')

# plt.subplot(1, 3, 2)
# plt.imshow(approx, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Approximation')

# plt.subplot(1, 3, 3)
# plt.imshow((approx - truth) ** 2)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Error')

# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.savefig(path_image_test31 + '.png')


# resolution = 61
# data = test_loader61.dataset[0]
# coeff = data.coeff.numpy().reshape((resolution, resolution))
# truth = data.y.numpy().reshape((resolution, resolution))
# approx = u_normalizer.decode(model(data).reshape(1,-1)).detach().numpy().reshape((resolution, resolution))
# _min = np.min(np.min(truth))
# _max = np.max(np.max(truth))

# # plt.figure()
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(truth, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Ground Truth')

# plt.subplot(1, 3, 2)
# plt.imshow(approx, vmin = _min, vmax=_max)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Approximation')

# plt.subplot(1, 3, 3)
# plt.imshow((approx - truth) ** 2)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.title('Error')

# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.savefig(path_image_test61 + '.png')
