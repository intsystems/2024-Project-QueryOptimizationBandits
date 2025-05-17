import json
import numpy as np
import torch
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
import net
from net import features, left_child, right_child
from featurize import TreeFeaturizer

from TreeConvolution.util import prepare_trees

import sys

sys.path.insert(0, "./bayes_deep_compression")

from src.methods.bayes.variational.net import NormalVarBayesLayer, VarBayesNet
from src.methods.bayes.variational.optimization import VarRenuiLoss
from src.methods.bayes.variational.net_distribution import VarBayesModuleNetDistribution

CUDA = False and torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _bayes_nn_path(base):
    return os.path.join(base, "bayes_nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")

def _inv_log1p(x):
    return np.exp(x) - 1

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(np.array(targets))
    return trees, targets

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose

        self.__bayes_net = None

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()

        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path):
        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)

        self.__net.load_state_dict(torch.load(_nn_path(path)), strict=False)
        #self.__net.eval()

        #lin_layer_1 = NormalVarBayesLayer(self.__net.tree_conv[-3])
        lin_layer_2 = NormalVarBayesLayer(self.__net.tree_conv[-1])

        #self.__bayes_net = VarBayesNet(self.__net.tree_conv, torch.nn.ModuleDict({'9': lin_layer_1, '11': lin_layer_2}))
        self.__bayes_net = VarBayesNet(self.__net.tree_conv, torch.nn.ModuleDict({'11': lin_layer_2}))

        #print(self.__bayes_net.state_dict())

        self.__bayes_net.load_state_dict(torch.load(_bayes_nn_path(path)))

        #self.__net_distribution = VarBayesModuleNetDistribution(self.__bayes_net.base_module, self.__bayes_net.posterior)
        #self.__net_distribution.set_mean_params()

        self.__bayes_net.sample()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path):
        # try to create a directory here
        os.makedirs(path, exist_ok=True)

        torch.save(self.__bayes_net.state_dict(), _bayes_nn_path(path))
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y, lr=0.001, beta=0.01, max_epoch=100):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        pairs = list(zip(X, y))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)

        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)

        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        #lin_layer_1 = NormalVarBayesLayer(self.__net.tree_conv[-3])
        lin_layer_2 = NormalVarBayesLayer(self.__net.tree_conv[-1])

        #self.__bayes_net = VarBayesNet(self.__net.tree_conv, torch.nn.ModuleDict({'9': lin_layer_1, '11': lin_layer_2}))
        self.__bayes_net = VarBayesNet(self.__net.tree_conv, torch.nn.ModuleDict({'11': lin_layer_2}))

        if CUDA:
            self.__bayes_net = self.__bayes_net.cuda()

        optimizer = torch.optim.Adam(self.__bayes_net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        bayes_loss_fn = VarRenuiLoss()
        
        losses = []
        for epoch in range(max_epoch):
            loss_accum = 0
            fit_losses = []
            dist_losses = []
            for x, y in dataset:
                if CUDA:
                    y = y.cuda()
                
                sample = self.__bayes_net.sample()
                
                trees = prepare_trees(x, features, left_child, right_child, CUDA)

                y_pred = self.__bayes_net(trees)

                fit_losses.append(loss_fn(y_pred, y))

                loss_accum += fit_losses[-1].item()

                dist_losses.append(bayes_loss_fn(posterior=self.__bayes_net.posterior, prior=self.__bayes_net.prior, param_sample_dict=sample))

            aggregation_result = bayes_loss_fn.aggregate(fit_losses, dist_losses, beta) 
            out = aggregation_result.total_loss
            
    
            optimizer.zero_grad()
            out.backward()
            optimizer.step()
            loss_accum /= len(fit_losses)
            losses.append(out.item())
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum, "cross loss:", losses[-1])

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")
        self.__bayes_net.sample()
        #self.__net_distribution = VarBayesModuleNetDistribution(self.__bayes_net.base_module, self.__bayes_net.posterior)
        #self.__net_distribution.set_mean_params()
        

    def predict(self, X):
        if not isinstance(X, (list, np.ndarray)):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        #self.__bayes_net.sample()

        X = self.__tree_transform.transform(X)
        
        pred = self.__bayes_net(prepare_trees(X, features, left_child, right_child, CUDA)).cpu().detach().numpy()
        return self.__pipeline.inverse_transform(pred)

