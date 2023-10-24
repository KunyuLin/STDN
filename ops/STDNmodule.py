import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class SpatialGroupingModule(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_sg, num_cg, tau):
        super(SpatialGroupingModule, self).__init__()
        assert dim_in % num_cg == 0
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_sg = num_sg    # spatial group
        self.num_cg = num_cg    # channel group (#TODO)
        self.tau = tau
        self.wc = torch.nn.Sequential(
                  torch.nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(dim_out, num_sg * num_cg * num_cg, kernel_size=3, padding=1),
                  )

    def forward(self, x):
        # postfix ``s'' indicates spatial, ``c'' indicates center
        # get spatial features
        zs = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2).contiguous()            # bz x 49 x D
        zs = zs.view(zs.shape[0], zs.shape[1], self.num_cg, self.dim_in//self.num_cg)   # bz x 49 x cg x D//cg
        zs = zs.view(zs.shape[0], zs.shape[1]*self.num_cg, self.dim_in//self.num_cg)    # bz x 49*cg x D//cg
        # get weights for centers
        wc = self.wc(x)                                                                 # bz x sg*cg*cg x 7 x 7
        wc = wc.view(wc.shape[0], wc.shape[1], -1)                                      # bz x sg*cg*cg x 49
        wc = wc.view(wc.shape[0], wc.shape[1]//self.num_cg, self.num_cg*wc.shape[-1])   # bz x sg*cg x cg*49
        wc = F.softmax(wc, dim=2)
        # get centers
        c = torch.bmm(wc, zs)                                                           # bz x sg*cg x D//cg
        # get weights for spatial features
        ws = torch.cdist(zs, c)                                                         # bz x 49*cg x sg*cg
        ws = torch.softmax(-ws/self.tau, dim=2)                     
        ws_sum = ws.sum(dim=1, keepdim=True)
        ws = ws / (ws_sum.detach()+1e-6)
        zc = torch.bmm(ws.transpose(1,2), zs)                                           # bz x sg*cg x D//cg
        # zc = zc.view(zc.shape[0], -1)                                                   # bz x sg*D
        return zc, ws

class SpatialRelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, feature_dim, img_feature_dim, num_group):
        super(SpatialRelationModuleMultiScale, self).__init__()
        self.subsample_num = 12 # how many relations selected to sum up
        self.feature_dim = feature_dim
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_group, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_group, scale)
            self.relations_scales.append(relations_scale)
            # print('Scale-{}, the number of permutations={}\n\tpermutations:'.format(scale, len(relations_scale)), relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_group = num_group
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Linear(scale * self.feature_dim, self.img_feature_dim)
            self.fc_fusion_scales += [fc_fusion]

        # print('Multi-Scale Spatial Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        act_all = []
        for scaleID in range(0, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            act_relation_list = []
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation_list.append(act_relation)
            act_relation = torch.stack(act_relation_list, dim=1).mean(dim=1)
            act_all.append(act_relation)
        act_all = torch.cat(act_all, dim=1)
        return act_all

    def return_relationset(self, num_group, num_group_relation):
        import itertools
        return list(itertools.permutations([i for i in range(num_group)], num_group_relation))

class Adapter(torch.nn.Module):
    def __init__(self, D_features, D_hidden_features):
        super().__init__()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features, bias=False)
        self.act = F.relu
        self.D_fc2 = nn.Linear(D_hidden_features, D_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = self.sigmoid(xs)
        x = x * xs
        return x

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, feature_dim, img_feature_dim, local_feature_dim, global_feature_dim, num_frames, num_class, dropout, 
                 num_spatial_group, num_channel_group, sgm_temp, aggregator_dim):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.dropout = dropout
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations
        self.num_spatial_group = num_spatial_group
        self.num_channel_group = num_channel_group
        # feature dim
        self.feature_dim = feature_dim
        self.img_feature_dim = img_feature_dim
        self.local_feature_dim = local_feature_dim
        self.global_feature_dim = global_feature_dim
        self.mixed_feature_dim = self.local_feature_dim * (self.num_spatial_group-1) + self.global_feature_dim
        self.aggregator_dim = aggregator_dim

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        self.num_bottleneck = 256
        self.proj_scales = nn.ModuleList()          # high-tech modulelist
        self.conv_scales = nn.ModuleList()          # high-tech modulelist
        self.fc_fusion_scales = nn.ModuleList()     # high-tech modulelist
        self.srn_scales = nn.ModuleList()
        self.adapter_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]
            proj = SpatialGroupingModule(self.feature_dim, self.img_feature_dim, num_spatial_group, num_channel_group, sgm_temp)
            srn = SpatialRelationModuleMultiScale(self.feature_dim, self.local_feature_dim, num_spatial_group*num_channel_group)
            conv = nn.Conv2d(self.feature_dim, self.global_feature_dim, kernel_size=7, stride=1, padding=0, bias=True)
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(p=self.dropout),
                        nn.Linear(scale * (self.local_feature_dim * (self.num_spatial_group-1) + self.global_feature_dim), self.num_bottleneck),
                        nn.ReLU(),
                        )
            adapter = Adapter(self.num_bottleneck, self.aggregator_dim)

            self.proj_scales += [proj]
            self.conv_scales += [conv]
            self.srn_scales += [srn]
            self.fc_fusion_scales += [fc_fusion]
            self.adapter_scales += [adapter]
        self.classifier = nn.Linear(self.num_bottleneck, self.num_class)

        # print('Multi-Scale Temporal Relation Network Module in use', ['%d-group relation' % i for i in self.scales])

    def forward(self, input):
        bz, T, _, _, _ = input.shape
        # the first one is the largest scale
        act_all_list = []
        input_ = input.view((-1,)+input.shape[2:])
        input_l, attn = self.proj_scales[0](input_)
        attn_list = [attn]
        input_l = self.srn_scales[0](input_l)
        input_g = self.conv_scales[0](input_).squeeze(-1).squeeze(-1)
        input_ = torch.cat((input_l, input_g), dim=1)
        input_ = input_.view((bz, T)+input_.shape[1:])
        act_relation = input_[:, self.relations_scales[0][0] , :]
        act_relation = act_relation.view((-1,)+act_relation.shape[2:])
        act_relation = act_relation.view(input.shape[0], -1, act_relation.shape[1])
        act_relation = act_relation.view(act_relation.size(0), self.scales[0] * self.mixed_feature_dim)
        act_relation_tot = self.fc_fusion_scales[0](act_relation)*self.subsample_num
        act_relations = [act_relation_tot]
        # adapter
        act_relation_tot = self.adapter_scales[0](act_relation_tot)
        act_all = self.classifier(act_relation_tot)
        act_all_list.append(act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            act_relation_tot = torch.zeros(input.shape[0], self.num_bottleneck).cuda()
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            input_ = input.view((-1,)+input.shape[2:])
            input_l, attn = self.proj_scales[scaleID](input_)
            attn_list.append(attn)
            input_l = self.srn_scales[scaleID](input_l)
            input_g = self.conv_scales[scaleID](input_).squeeze(-1).squeeze(-1)
            input_ = torch.cat((input_l, input_g), dim=1)
            input_ = input_.view((bz, T)+input_.shape[1:])
            for idx in idx_relations_randomsample:
                act_relation = input_[:, self.relations_scales[scaleID][idx] , :]
                act_relation = act_relation.view((-1,)+act_relation.shape[2:])
                act_relation = act_relation.view(input.shape[0], -1, act_relation.shape[1])
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.mixed_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation_tot += act_relation
            act_relations.append(act_relation_tot)
            # adapter
            act_relation_tot = self.adapter_scales[scaleID](act_relation_tot)
            act_all = self.classifier(act_relation_tot)
            act_all_list.append(act_all)
        return torch.stack(act_all_list, dim=1), torch.stack(act_relations, dim=1), attn_list

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


def return_STDN(relation_type, feature_dim, img_feature_dim, local_feature_dim, global_feature_dim, 
               num_frames, num_class, dropout, num_spatial_group, num_channel_group, sgm_temp, aggregator_dim):
    if relation_type == 'STDN':
        model = RelationModuleMultiScale(feature_dim, img_feature_dim, local_feature_dim, global_feature_dim, 
                                            num_frames, num_class, dropout, num_spatial_group, num_channel_group, sgm_temp, aggregator_dim)
    else:
        raise ValueError('Unknown' + relation_type)


    return model

