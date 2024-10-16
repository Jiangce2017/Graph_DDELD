import torch
import numpy as np
import wandb
import sys
import os.path as osp

import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from utilities import *
from nn_conv import NNConv_old

from timeit import default_timer
from find_neighbors_model import Neihbor_finder
from training_functions_graph import train, test

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
    

## Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./darcy.yaml", config_name="default", config_folder="./config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="./config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

## Set up WandB logging
if config.wandb.log:
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.scope,
                config.nn_type,
                config.width,
                config.window_size,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)
else: 
    wandb_init_args = None

## Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

ntrain = config.ntrain
ntest = config.ntest
r = config.r
s = int(((241 - 1)/r) + 1)
n = s**2
m = config.m
k = config.k

## import dataset
reader = MatReader(config.train_data_path)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u64 = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(config.test_data_path)
test_a = reader.read_field('coeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::4,::4].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::4,::4].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::4,::4].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::4,::4].reshape(ntest,-1)

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

train_a31 =train_a[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_smooth31 = train_a_smooth[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_gradx31 = train_a_gradx[:ntrain,::2,::2].reshape(ntrain,-1)
train_a_grady31 = train_a_grady[:ntrain,::2,::2].reshape(ntrain,-1)
train_u31 = train_u[:ntrain,::2,::2].reshape(ntrain,-1)

test_a = a_normalizer.encode(test_a)
test_a_smooth = as_normalizer.encode(test_a_smooth)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
test_a_grady = agy_normalizer.encode(test_a_grady)

test_a = test_a.reshape(ntest,61,61)
test_a_smooth = test_a_smooth.reshape(ntest,61,61)
test_a_gradx = test_a_gradx.reshape(ntest,61,61)
test_a_grady = test_a_grady.reshape(ntest,61,61)
test_u = test_u.reshape(ntest,61,61)

test_a31 =test_a[:ntest,::2,::2].reshape(ntest,-1)
test_a_smooth31 = test_a_smooth[:ntest,::2,::2].reshape(ntest,-1)
test_a_gradx31 = test_a_gradx[:ntest,::2,::2].reshape(ntest,-1)
test_a_grady31 = test_a_grady[:ntest,::2,::2].reshape(ntest,-1)
test_u31 = test_u[:ntest,::2,::2].reshape(ntest,-1)

## preprocessing dataset
u_normalizer = GaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[31,31])
edge_index = meshgenerator.ball_connectivity(config.radius_train)
grid = meshgenerator.get_grid()
data_train31 = []
for j in range(ntrain):
    edge_attr = meshgenerator.attributes(theta=train_a31[j,:])
    data_train31.append(Data(x=torch.cat([grid, train_a31[j,:].reshape(-1, 1),
                                       train_a_smooth31[j,:].reshape(-1, 1), train_a_gradx31[j,:].reshape(-1, 1), train_a_grady31[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=train_u31[j, :], coeff=train_a31[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                          ))
print('31 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[31,31])
edge_index = meshgenerator.ball_connectivity(config.radius_test)
grid = meshgenerator.get_grid()
data_test31 = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a31[j,:])
    data_test31.append(Data(x=torch.cat([grid, test_a31[j,:].reshape(-1, 1),
                                       test_a_smooth31[j,:].reshape(-1, 1), test_a_gradx31[j,:].reshape(-1, 1), test_a_grady31[j,:].reshape(-1, 1)
                                       ], dim=1),
                           y=test_u31[j, :], coeff=test_a31[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                          ))
print('31 grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

train_loader = DataLoader(data_train31, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(data_test31, batch_size=config.batch_size, shuffle=False)

## set device
device = torch.device(config.device)

## setup model
model = KernelNN(config.width,config.ker_width,config.depth,config.edge_features,config.node_features)
model = model.to(device)

## setup optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
cur_ep = 0

myloss = LpLoss(size_average=False)
u_normalizer = u_normalizer.to(device)

## setup neighbor finder
data_iterator = iter(train_loader)
data_batch = next(data_iterator)
batch, x, edge_index, edge_attr = data_batch.batch, data_batch.x, data_batch.edge_index, data_batch.edge_attr
num_node = x.shape[0]//config.batch_size
num_edge = edge_index.shape[1]//config.batch_size
print("single graph num_node: {}".format(num_node))
print("single graph num_edge: {}".format(num_edge))
print("batch number: {}".format(config.batch_size))
edge_index_single = edge_index[:,edge_index[0,:]<num_node]
finder = Neihbor_finder(num_node,num_edge, edge_index_single,config.num_subgraph_batch, config.batch_size)
full_node_idx = torch.randperm(num_node) 
neighbor_centers_list = torch.split(full_node_idx,config.num_subgraph_batch,dim=0)
config['num_node'] = num_node
config['num_edge'] = num_edge

## setup logger
exp_name = config.nn_type + '_' + config.scope + '_'  + str(config.ntrain)+ '_' + str(config.ntest)+'_'
if config.scope == 'local':
    exp_name += str(config.num_hops) + '_'
print(exp_name) 
train_logger = Logger(
    osp.join(config.results_dir, exp_name+'train.log'),
    ['ep', 'train_mse','train_l2','train_r2']
)
test_logger = Logger(
    osp.join(config.results_dir, exp_name+'valid.log'),
    ['ep', 'test_mse','test_l2','test_r2']
)


## train the model
for ep in range(cur_ep+1, config.epochs+1):
    start_time = default_timer()
    train_mse, train_l2, train_r2 = train(model,neighbor_centers_list,finder,optimizer,scheduler,train_loader,device,u_normalizer,config)
    end_time = default_timer()
    epoch_time = end_time - start_time
    print('Epoch {}, time {:.4f}'.format(ep, epoch_time))
    print('train_mse: {:.8f},train_l2: {:.8f}, train_r2: {:.8f}'.format(train_mse,train_l2, train_r2))

    if ep % config.eval_epoch == 0:
        test_mse, test_l2, test_r2 = test(model,neighbor_centers_list,finder,test_loader,device,u_normalizer,config)
        print('test_mse: {:.8f},test_l2: {:.8f}, test_r2: {:.8f}'.format(test_mse,test_l2, test_r2))
        train_logger.log({
            'ep': ep,             
            'train_mse': train_mse,
            'train_l2': train_l2,
            'train_r2': train_r2
        })
        test_logger.log({
            'ep': ep,             
            'test_mse': test_mse,
            'test_l2': test_l2,
            'test_r2': test_r2
        })    
        #torch.save(model.state_dict(), osp.join(config.mode_dir, exp_name+'model_ep{}.pth'.format(ep)))
        if config.wandb.log: 
            wandb.log({
                'train_mse': train_mse,
                'train_l2': train_l2,
                'train_r2': train_r2,
            })
            wandb.log({
                'test_mse': test_mse,
                'test_l2': test_l2,
                'test_r2': test_r2,
            })
            wandb.log_model(path=osp.join(config.results_dir, exp_name+'_model_ep{}.pth'.format(ep)), name=exp_name+'_model_ep{}'.format(ep))
## plotting results

