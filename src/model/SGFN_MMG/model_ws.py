import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import product

# from clip_adapter.model import AdapterModel
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMG import MMG_ws
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti)

# from src.model.model_utils.model_pointnetpp import PointNetPP

from src.utils.eva_utils_acc import (evaluate_topk_object, evaluate_topk_object_w_backup, 
                                 evaluate_topk_predicate,
                                 evaluate_triplet_topk, get_gt)
from utils import op_utils
from scipy.optimize import linear_sum_assignment
from src.utils.eval_utils_recall import evaluate_triplet_recallk, evaluate_triplet_mrecallk



class Mmgnet(BaseModel):
    def __init__(self, config, classNames, relationNames, dim_descriptor=11):
        '''
        3d cat location, 2d
        '''
        
        super().__init__('Mmgnet', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        assert (mconfig.use_pair_match and mconfig.use_pair_loss) != True

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel

        self.classNames = classNames
        self.relationNames = relationNames
        self.num_class = len(classNames)
        self.num_rel = len(relationNames)
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim
        self.object_dim = 512
        self.momentum = 0.1
        self.model_pre = None
        
        self.obj_3d_encoder = PointNetfeat(
            device = config.DEVICE,
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=self.object_dim)    
        

        self.box_feature_mapping = torch.nn.Sequential(
            torch.nn.Linear(8, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

        self.clip, preprocess = clip.load("ViT-B/32", device=self.config.DEVICE)  

        obj_text_input = op_utils.obj_prompt(classNames)
        texts = clip.tokenize(obj_text_input).to(self.config.DEVICE)
        self.obj_text_clip_fea = self.clip.encode_text(texts)
        self.obj_text_clip_fea = self.obj_text_clip_fea / self.obj_text_clip_fea.norm(dim=1, keepdim=True)


        # Relationship Encoder
        self.rel_encoder_3d = PointNetfeat(
            device = config.DEVICE,
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        
        self.mmg = MMG_ws(
            dim_node=512,
            dim_edge=512,
            dim_atten=self.mconfig.DIM_ATTEN,
            cuda_device=self.config.DEVICE, 
            depth=self.mconfig.N_LAYERS, 
            num_heads=self.mconfig.NUM_HEADS,
            aggr=self.mconfig.GCN_AGGR,
            flow=self.flow,
            attention=self.mconfig.ATTENTION,
            use_edge=self.mconfig.USE_GCN_EDGE,
            DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)

        self.triplet_projector_3d = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_class + self.num_rel)
        )

        # object adapter
        self.obj_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )


        self.relation_map = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        
        if mconfig.multi_rel_outputs:
            self.rel_predictor_3d = PointNetRelClsMulti(
                self.num_rel, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            self.rel_predictor_2d = PointNetRelClsMulti(
                self.num_rel, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
        else:
            self.rel_predictor_3d = PointNetRelCls(
                self.num_rel, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            self.rel_predictor_2d = PointNetRelCls(
                self.num_rel, 
                in_size=512, 
                batch_norm=with_bn,drop_out=True)
            
        self.init_weight()
        
        mmg_obj, mmg_rel = [], []
        for name, para in self.mmg.named_parameters():
            if 'nn_edge' in name:
                mmg_rel.append(para)
            else:
                mmg_obj.append(para)
        
        self.edge_weight = torch.nn.Parameter(torch.randn(512, 512))
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_3d_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder_3d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_obj, 'lr':float(config.LR) / 4, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':mmg_rel, 'lr':float(config.LR) / 2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor_3d.parameters(), 'lr':float(config.LR) / 10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.box_feature_mapping.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.triplet_projector_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_logit_scale, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.relation_map.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.mlp_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[0].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[-1].weight)
        torch.nn.init.xavier_uniform_(self.triplet_projector_3d[3].weight)
        obj_text_features, _ = self.get_label_weight()

        # 2d node classifier
        self.obj_predictor_2d = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_2d.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_2d.parameters():
            param.requires_grad = True
            
        # node feature classifier        
        self.obj_predictor_3d = torch.nn.Linear(self.mconfig.clip_feat_dim, self.num_class)
        self.obj_predictor_3d.weight.data.copy_(obj_text_features)
        for param in self.obj_predictor_3d.parameters():
            param.requires_grad = True

        # freeze clip adapter
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.obj_logit_scale.requires_grad = True
    
    def update_model_pre(self, new_model):
        self.model_pre = new_model
    
    def get_label_weight(self):
        self.clip_model, preprocess = clip.load("ViT-B/32", device=self.config.DEVICE)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # get norm clip weight
        obj_prompt = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.classNames]).cuda(self.config.DEVICE)
        rel_prompt = torch.cat([clip.tokenize(f"{c}") for c in self.relationNames]).cuda(self.config.DEVICE)

        with torch.no_grad():
            obj_text_features = self.clip_model.encode_text(obj_prompt)
            rel_text_features = self.clip_model.encode_text(rel_prompt)
        
        obj_text_features = obj_text_features / obj_text_features.norm(dim=-1, keepdim=True)
        rel_text_features = rel_text_features / rel_text_features.norm(dim=-1, keepdim=True)

        return obj_text_features.float(), rel_text_features.float()

    def cosine_loss(self, A, B, t=1, weight = None):
        if weight == None:
            return torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0).mean()
        else:
            simi = torch.clamp(t - F.cosine_similarity(A, B, dim=-1), min=0)
            simi = torch.matmul(simi, weight) / len(simi)
            return simi

    
    def contrastive_loss(self, A, B, batch_ids, t=1):
        split_batch_ids = [sum(batch_ids == i) for i in batch_ids.unique()]
        A_list = A.split(split_batch_ids, dim=0)
        B_list = B.split(split_batch_ids, dim=0)

        softmax = torch.nn.LogSoftmax(dim=-1)
        loss = torch.tensor(0.0).cuda(self.config.DEVICE)

        for a, b in zip(A_list, B_list):
            r = torch.mm(a, b.T) / t
            log_r = softmax(r)
            loss -= torch.sum(torch.diag(log_r)) / r.shape[0]
        loss = loss / len(A_list)
        return loss


    def get_object_label_hungarain_plus(self, obj_2d_feature, obj_texts, batch_ids):
        with torch.no_grad():
            obj_label = []
            batch_ids = batch_ids.squeeze(-1)
            for i, obj_text in enumerate(obj_texts):
                obj_text = np.unique(obj_text)
                obj_text_idx = torch.tensor([self.classNames.index(i) for i in obj_text]).cuda(self.config.DEVICE)
                
                obj_text_feature = self.obj_text_clip_fea[obj_text_idx]
                
                obj_2d_fea = obj_2d_feature[batch_ids == i]
                Num_2d = obj_2d_fea.shape[0]
                obj_2d_fea = obj_2d_fea / obj_2d_fea.norm(dim=1, keepdim=True)

                logit_scale = self.clip.logit_scale.exp()
            
                logits_per_text = logit_scale * obj_text_feature @ obj_2d_fea.t()
                row_ind, col_ind = linear_sum_assignment(-logits_per_text.cpu())

                f_indexs = torch.zeros(Num_2d, dtype = int).cuda(self.config.DEVICE)
                f_indexs[col_ind] = obj_text_idx
                
                no_match_2d_idx = torch.tensor([i for i in range(Num_2d) if i not in col_ind]).cuda(self.config.DEVICE)
                if no_match_2d_idx.shape[0] != 0:
                    no_match_2d_fea = obj_2d_fea[no_match_2d_idx]
                    
                    logits_per_image = logit_scale * no_match_2d_fea @ obj_text_feature.t()
                    col_ind = torch.max(logits_per_image, dim=1)[1]
                    obj_indexs = obj_text_idx[col_ind]
                    f_indexs[no_match_2d_idx] = obj_indexs

                obj_label.append(f_indexs)
        return torch.hstack(obj_label).cuda(self.config.DEVICE)



    def get_object_label(self, obj_2d_feature, obj_texts, batch_ids):
        with torch.no_grad():
            obj_label = []
            batch_ids = batch_ids.squeeze(-1)
            for i, obj_text in enumerate(obj_texts):
                obj_text = np.unique(obj_text)
                obj_text_idx = torch.tensor([self.classNames.index(i) for i in obj_text]).cuda(self.config.DEVICE)
                obj_text_feature = self.obj_text_clip_fea[obj_text_idx]
                
                obj_2d_fea = obj_2d_feature[batch_ids == i]
                obj_2d_fea = obj_2d_fea / obj_2d_fea.norm(dim=1, keepdim=True)
            
                logits_per_image = self.clip.logit_scale.exp() * obj_2d_fea @ obj_text_feature.t()
                row_ind, col_ind = linear_sum_assignment(-logits_per_image.cpu())
                
                obj_indexs = obj_text[col_ind]
                f_indexs = torch.tensor([self.classNames.index(i) for i in obj_indexs])
                obj_label.append(f_indexs)
        return torch.hstack(obj_label).cuda(self.config.DEVICE)
    

    def get_object_label_wo_hungarain(self, obj_2d_feature, obj_texts, batch_ids):
        with torch.no_grad():
            obj_label = []
            batch_ids = batch_ids.squeeze(-1)
            for i, obj_text in enumerate(obj_texts):
                obj_text = np.unique(obj_text)
                obj_text_idx = torch.tensor([self.classNames.index(i) for i in obj_text]).cuda(self.config.DEVICE)
                
                obj_text_feature = self.obj_text_clip_fea[obj_text_idx]
                
                obj_2d_fea = obj_2d_feature[batch_ids == i]
                obj_2d_fea = obj_2d_fea / obj_2d_fea.norm(dim=1, keepdim=True)
            
                logits_per_image = self.clip.logit_scale.exp() * obj_2d_fea @ obj_text_feature.t()
                col_ind = torch.max(logits_per_image, dim=1)[1].cpu()
                
                obj_indexs = obj_text[col_ind]
                f_indexs = torch.tensor([self.classNames.index(i) for i in obj_indexs])
                obj_label.append(f_indexs)
        return torch.hstack(obj_label).cuda(self.config.DEVICE)

        

    def get_best_match(self, tri_label, tri_token, sub_fea, obj_fea, rel_fea, edge_idx, edge_temp, temp_rel_lable, topk, descriptor, img_pair_info_temp):
        def get_match(fea_0, fea_1):
            fea_0 = fea_0 / fea_0.norm(dim=-1, keepdim=True)
            fea_1 = fea_1 / fea_1.norm(dim=-1, keepdim=True)
            logit_scale = self.clip.logit_scale.detach().clone().exp()

            logits_per_fea_0 = logit_scale * fea_0 @ fea_1.t()
            return torch.sigmoid(logits_per_fea_0).squeeze(0)
        
        with torch.no_grad():
            tri_fea = self.clip.encode_text(tri_token)  # 文本特征
            shape_weight = None

            if self.mconfig.use_shape_trick:
                if tri_label[1] in ["bigger than", "smaller than", "higher than", "lower than"]:
                    shape_weight = op_utils.shape_trick(tri_label, descriptor, edge_temp)

            if rel_fea is None:
                sub_logits = get_match(tri_fea, sub_fea.half())
                obj_logits = get_match(tri_fea, obj_fea.half())
                triplet_logits = torch.hstack([(sub_logits[0][i] + obj_logits[0][j]) / 2 for i, j in edge_temp])
            else:
                triplet_fea = []
                for i in range(len(edge_idx)):
                    if self.mconfig.triplet_fea_get_way == "mean":
                        t = (sub_fea[edge_temp[i][0]] + rel_fea[i] + obj_fea[edge_temp[i][1]]) / 3
                    if self.mconfig.triplet_fea_get_way == "max":
                        t = torch.max(torch.vstack([sub_fea[edge_temp[i][0]], rel_fea[i], obj_fea[edge_temp[i][1]]]), dim=0)[0]
                    if self.mconfig.triplet_fea_get_way == "only_rel":
                        t = rel_fea[i]
                    if len(img_pair_info_temp)!= 0:
                        if len(img_pair_info_temp[i]) != 0:
                            t = img_pair_info_temp[i]  # Only extract the features of the pair of images.
                            # t = (t + img_pair_info_temp[i]) / 2
                    triplet_fea.append(t)
                triplet_fea = torch.vstack(triplet_fea).half()
                triplet_logits = get_match(tri_fea, triplet_fea).squeeze(0)

            alpha = 1.0
            
            if shape_weight is not None:
                triplet_logits = triplet_logits + alpha * shape_weight

        tri_index = torch.sort(triplet_logits)[1]
        for k, idx in enumerate(tri_index):
            if k == topk:
                break
            i = edge_idx[idx]
            temp_rel_lable[i][self.relationNames.index(tri_label[1])] = 1.0

        return temp_rel_lable
    
    
    def get_rel_label(self, obj_label, tri_texts, batch_ids, obj_features, relation_fea = None, descriptor = None, img_pair_info=None, img_pair_idx=None, edge_indices_all=None):
        """Match and confirm the features of the relation using the features of the triplet text and the three-dimensional features of the subject and object."""
        def change_array_to_dict(tri_texts):
            tri_dict = {}
            for key in tri_texts:
                k = tuple(key)
                if k not in tri_dict.keys():
                    tri_dict[k] = 1
                else:
                    tri_dict[k] += 1
            return tri_dict
        
        split_batch_ids = [sum(batch_ids == i) for i in batch_ids.unique()]
        
        rel_label_list = []

        edge_init_idx = 0
        for i in range(len(tri_texts)):
            idx = (batch_ids == i).squeeze(-1)
            obj_feature_temp = obj_features[idx]
            
            descriptor_temp = descriptor[idx]
            obj_label_i = obj_label[idx]
            
            edge_indices = list(product(list(range(split_batch_ids[i])), list(range(split_batch_ids[i]))))
            edge_indices = [k for k in edge_indices if k[0]!=k[1]]
            L = len(edge_indices)

            if relation_fea is not None:
                rel_feature_temp = relation_fea[edge_init_idx : edge_init_idx + L]

            temp_rel_lable = [torch.zeros(self.num_rel).cuda(self.config.DEVICE) for _ in range(L)]
            
            tri_dict = change_array_to_dict(tri_texts[i])

            for tri, num in tri_dict.items():
                sub_idx = self.classNames.index(tri[0])
                obj_idx = self.classNames.index(tri[2])
                
                sub_fea_idx = torch.where(obj_label_i == sub_idx)[0]
                obj_fea_idx = torch.where(obj_label_i == obj_idx)[0]

                if len(sub_fea_idx) == 0 or len(obj_fea_idx) == 0:  
                    continue

                if len(sub_fea_idx) == len(obj_fea_idx) == 1:
                    if tri[0] != tri[2]:
                        temp_rel_lable[edge_indices.index((int(sub_fea_idx[0]), int(obj_fea_idx[0])))][self.relationNames.index(tri[1])] = 1.0
                else:
                    edge_sub_obj = list(product(list(sub_fea_idx), list(obj_fea_idx)))
                    edge_sub_obj = [i for i in edge_sub_obj if i[0] != i[1]]
                    edge_idx = [edge_indices.index(i) for i in edge_sub_obj]
                    
                    quan_edge_idx = [t_i + edge_init_idx for t_i in edge_idx]
                    if self.mconfig.use_pair_match:
                        img_pair_info_temp = [img_pair_info[img_pair_idx == t_idx].squeeze(0) for t_idx in quan_edge_idx]
                    else:
                        img_pair_info_temp = []

                    edge_temp = list(product(list(range(len(sub_fea_idx))), list(range(len(obj_fea_idx)))))
                    if tri[0] == tri[2]:
                        edge_temp = [i for i in edge_temp if i[0]!=i[1]]

                    tri_token = clip.tokenize(op_utils.tri_prompt(tri)).to(self.config.DEVICE)
                    
                    sub_fea = obj_feature_temp[sub_fea_idx]
                    obj_fea = obj_feature_temp[obj_fea_idx]

                    descriptor_list = [descriptor_temp[sub_fea_idx], descriptor_temp[obj_fea_idx]]

                    if relation_fea is not None:
                        rel_fea = rel_feature_temp[edge_idx]
                    else:
                        rel_fea = None

                    temp_rel_lable = self.get_best_match(tri, tri_token, sub_fea, obj_fea, rel_fea, edge_idx, edge_temp, temp_rel_lable, num, descriptor_list, img_pair_info_temp)

            rel_label_list += temp_rel_lable
            edge_init_idx += L
        
        rel_label_list = torch.vstack(rel_label_list).cuda(self.config.DEVICE)

        return rel_label_list
    
    
    def get_rel_label_no_mask(self, tri_texts, batch_ids, obj_features, relation_fea = None, distance_weight = None, descriptor = None, img_pair_info=None, img_pair_idx=None):
        """Match and confirm the features of the relation using the features of the triplet text and the three-dimensional features of the subject and object."""
        def change_array_to_dict(tri_texts):
            tri_dict = {}
            for key in tri_texts:
                k = tuple(key)
                if k not in tri_dict.keys():
                    tri_dict[k] = 1
                else:
                    tri_dict[k] += 1
            return tri_dict
        
        split_batch_ids = [sum(batch_ids == i) for i in batch_ids.unique()]
        rel_label_list = []
        edge_init_idx = 0
        for i in range(len(tri_texts)):
            idx = (batch_ids == i).squeeze(-1)
            obj_feature_temp = obj_features[idx]
            
            descriptor_temp = descriptor[idx]
            
            edge_indices = list(product(list(range(split_batch_ids[i])), list(range(split_batch_ids[i]))))
            edge_indices = [i for i in edge_indices if i[0]!=i[1]]
            L = len(edge_indices)

            if relation_fea is not None:
                rel_feature_temp = relation_fea[edge_init_idx : edge_init_idx + L]

            temp_rel_lable = [torch.zeros(self.num_rel).cuda(self.config.DEVICE) for _ in range(L)]
            
            tri_dict = change_array_to_dict(tri_texts[i])

            edge_idx = [edge_indices.index(i) for i in edge_indices]

            if self.mconfig.use_pair_match:
                quan_edge_idx = [t_i + edge_init_idx for t_i in edge_idx]
                img_pair_info_temp = [img_pair_info[img_pair_idx == t_idx].squeeze(0) for t_idx in quan_edge_idx]
            else:
                img_pair_info_temp = []


            sub_fea = obj_feature_temp
            obj_fea = obj_feature_temp
            descriptor_list = [descriptor_temp, descriptor_temp]

            if relation_fea is not None:
                rel_fea = rel_feature_temp[edge_idx]
            else:
                rel_fea = None

            for tri, num in tri_dict.items():
                tri_token = clip.tokenize(op_utils.tri_prompt(tri)).to(self.config.DEVICE)
                temp_rel_lable = self.get_best_match(tri, tri_token, sub_fea, obj_fea, rel_fea, edge_idx, edge_indices, temp_rel_lable, num, descriptor_list, img_pair_info_temp)
            
            rel_label_list += temp_rel_lable
            edge_init_idx += L
        return torch.vstack(rel_label_list).cuda(self.config.DEVICE)
                    

    def get_tri_predict_and_label(self, tri_list):
        """Used to obtain the classification label for the triplet text."""
        tri_fea_list = []
        tri_label_list = []
        for tris in tri_list:
            tri_label = torch.zeros((len(tris), self.num_class + self.num_rel))
            tri_tokens = []
            for idx, i in enumerate(tris):
                tri_tokens.append(op_utils.tri_prompt(i))
                tri_label[idx][self.classNames.index(i[0])] = 1.0
                tri_label[idx][self.classNames.index(i[2])] = 1.0
                tri_label[idx][self.num_class + self.relationNames.index(i[1])] = 1.0
                           
            tri_tokens = clip.tokenize(tri_tokens).to(self.config.DEVICE)
            tri_fea = self.clip.encode_text(tri_tokens)
            tri_fea = self.triplet_projector_3d(tri_fea.float())  
            tri_fea = torch.sigmoid(tri_fea)
            tri_fea_list.append(tri_fea)
            tri_label_list.append(tri_label)
        return torch.vstack(tri_fea_list), torch.vstack(tri_label_list).cuda(self.config.DEVICE)
        

    def create_rnn_init_feature(self, edge_indices, entity_features, rel_features):
        sub_fea = entity_features[edge_indices[0]]
        obj_fea = entity_features[edge_indices[1]]
        return_fea = torch.stack([sub_fea, rel_features, obj_fea]).permute(1, 0, 2)
        return return_fea


    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids, gt_obj_labels, obj_texts = None, tri_texts = None, img_pair_info=None, img_pair_idx=None, istrain=False):
        
        obj_feature = self.obj_3d_encoder(obj_points)
        
        obj_feature = self.mlp_3d(obj_feature)

        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()  # descriptor：Center point coordinates, variance, difference between the coordinates of the maximum and minimum points, volume, and longest edge.
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            tmp_fea = self.box_feature_mapping(tmp)
            obj_feature = obj_feature + tmp_fea  
        
        if istrain:
            obj_feature_3d_mimic = obj_feature.clone()  # [B * num_obj_per_scan, 512]
        
        ''' Create edge feature '''
        with torch.no_grad():
           
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)  

        rel_feature_3d = self.rel_encoder_3d(edge_feature)  
        

        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_3d, gcn_edge_feature_3d, distance_weight \
            = self.mmg(obj_feature, rel_feature_3d, edge_indices, batch_ids, obj_center, descriptor.clone(), istrain=istrain)
        
        gcn_obj_feature_3d = gcn_obj_feature_3d + obj_feature

        if istrain:
            obj_features_2d_mimic = obj_2d_feats.clone()
            obj_3d_persudo_label = gt_obj_labels
            rel_3d_persudo_label = None
            
            if self.mconfig.use_object_pesudo_labels:
                if self.mconfig.use_hungarian:
                    if self.mconfig.use_object_num:
                        obj_3d_persudo_label = self.get_object_label(obj_2d_feats.detach().clone().half(), obj_texts, batch_ids)
                    else:
                        obj_3d_persudo_label = self.get_object_label_hungarain_plus(obj_2d_feats.detach().clone().half(), obj_texts, batch_ids)
                else:
                    obj_3d_persudo_label = self.get_object_label_wo_hungarain(obj_2d_feats.detach().clone().half(), obj_texts, batch_ids)

            
            if self.mconfig.rel_match_way == "triplet":
                t_rel = gcn_edge_feature_3d  # .detach().clone()
            else:
                t_rel = None

            if self.mconfig.use_relation_pesudo_labels:
                if self.mconfig.use_mask_filter:
                    rel_3d_persudo_label = self.get_rel_label(obj_3d_persudo_label, tri_texts, batch_ids, gcn_obj_feature_3d.detach().clone(),
                                                            t_rel, descriptor = descriptor, img_pair_info=img_pair_info, img_pair_idx=img_pair_idx, edge_indices_all=edge_indices)
                else:
                    rel_3d_persudo_label = self.get_rel_label_no_mask(tri_texts, batch_ids, gcn_obj_feature_3d.detach().clone(), 
                                                            t_rel, descriptor = descriptor, img_pair_info=img_pair_info, img_pair_idx=img_pair_idx)
        
        object_label_backup = None

        rel_cls_3d = self.rel_predictor_3d(gcn_edge_feature_3d)  

        logit_scale = self.obj_logit_scale.exp()
        obj_logits_3d = logit_scale * self.obj_predictor_3d(gcn_obj_feature_3d / gcn_obj_feature_3d.norm(dim=-1, keepdim=True))
        obj_logits_2d = obj_logits_3d

        if istrain:
            return obj_logits_3d, rel_cls_3d, obj_feature_3d_mimic, obj_features_2d_mimic, logit_scale, obj_3d_persudo_label, rel_3d_persudo_label, rel_feature_3d
        else:
            return obj_logits_3d, rel_cls_3d, obj_logits_2d, object_label_backup

    


    
    def process_train(self, obj_points, obj_2d_feats, descriptor, edge_indices, batch_ids, obj_texts, tri_texts, gt_rel_cls, gt_class, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None, img_pair_info=None, img_pair_idx=None):
        self.iteration +=1   

        obj_logits_3d, rel_cls_3d, obj_feature_3d, obj_feature_2d, obj_logit_scale, obj_persudo_label, rel_persudo_label, visual_rel_feature_3d = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, gt_class, obj_texts, tri_texts, img_pair_info=img_pair_info, img_pair_idx=img_pair_idx, istrain=True)  
        
        if self.mconfig.use_relation_pesudo_labels:
            pass
        else:
            rel_persudo_label = gt_rel_cls

        # compute loss for obj
        if self.mconfig.use_loss_obj:
            if self.mconfig.use_object_pesudo_labels:
                obj_label_one = F.one_hot(obj_persudo_label, self.num_class).float()
            else:
                obj_label_one = F.one_hot(gt_class, self.num_class).float()
            loss_obj_3d = F.cross_entropy(obj_logits_3d, obj_label_one) 

        # compute loss for rel
        if self.mconfig.use_loss_rel:
            if self.mconfig.multi_rel_outputs:
                if self.mconfig.WEIGHT_EDGE == 'BG':
                    if self.mconfig.w_bg != 0:
                        weight = self.mconfig.w_bg * (1 - rel_persudo_label) + (1 - self.mconfig.w_bg) * rel_persudo_label
                    else:
                        weight = None
                elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                    batch_mean = torch.sum(rel_persudo_label, dim=(0))  # [26]
                    zeros = (rel_persudo_label.sum(-1) ==0).sum().unsqueeze(0)  # Count the number of relations with the category "None".
                    batch_mean = torch.cat([zeros,batch_mean],dim=0)  # Record the number of each relation category in a batch.
                    weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                    if ignore_none_rel:
                        weight[0] = 0
                        weight *= 1e-2 # reduce the weight from ScanNet
                        # print('set weight of none to 0')
                    if 'NONE_RATIO' in self.mconfig:
                        weight[0] *= self.mconfig.NONE_RATIO
                        
                    weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                    weight = weight[1:]  # Exclude "None" relations.     
                elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                    weight = weights_rel
                elif self.mconfig.WEIGHT_EDGE == 'NONE':
                    weight = None
                else:
                    raise NotImplementedError("unknown weight_edge type")
                loss_rel_3d = F.binary_cross_entropy(rel_cls_3d, rel_persudo_label, weight=weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        if self.mconfig.use_loss_mimic:
            obj_feature_3d = obj_feature_3d / (obj_feature_3d.norm(dim=-1, keepdim=True) + 1e-8)
            obj_feature_2d = obj_feature_2d / (obj_feature_2d.norm(dim=-1, keepdim=True) + 1e-8)
            # loss_mimic = self.contrastive_loss(obj_feature_3d, obj_feature_2d, batch_ids, t=10)
            loss_mimic = self.cosine_loss(obj_feature_3d, obj_feature_2d)
        
        if self.mconfig.use_pair_loss and len(img_pair_idx) != 0:
            loss_pair = self.cosine_loss(visual_rel_feature_3d[img_pair_idx], img_pair_info)
        
        loss = torch.tensor(0.0).cuda(self.config.DEVICE)
        
        if self.mconfig.use_loss_obj:
            loss = loss + lambda_o * loss_obj_3d
        if self.mconfig.use_loss_rel:
            loss = loss + 3 * lambda_r * loss_rel_3d
        if self.mconfig.use_loss_mimic:
            loss = loss + 0.1 * loss_mimic
        if self.mconfig.use_pair_loss:
            loss = loss + loss_pair

        self.backward(loss)
        
        # compute 3d metric
        top_k_obj = evaluate_topk_object(obj_logits_3d.detach(), obj_persudo_label, topk=11)
        gt_edges = get_gt(obj_persudo_label, rel_persudo_label, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [ ("train/loss_obj_3d", loss_obj_3d.detach().item()),
                ("train/loss_rel_3d", loss_rel_3d.detach().item()),
                ("train/loss_mimic", loss_mimic.detach().item()),
                ("train/logit_scale", obj_logit_scale.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        log = []

        if self.mconfig.use_loss_obj:
            log.append(("train/loss_obj_3d", loss_obj_3d.detach().item()))
        if self.mconfig.use_loss_rel:
            log.append(("train/loss_rel_3d", loss_rel_3d.detach().item()))
        if self.mconfig.use_loss_mimic:
            log.append(("train/loss_mimic", loss_mimic.detach().item()))
        if self.mconfig.use_pair_loss:
            log.append(("train/loss_pair", loss_pair.detach().item()))
    
        log.append(("train/logit_scale", obj_logit_scale.detach().item()))
        log.append(("train/loss", loss.detach().item()))
        return log


    def process_val(self, result_print, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, scan_id=None, split_id=None, origin_obj_points=None, with_log=False, use_triplet=False, cal_recall=False):
 
        if not self.mconfig.random_validation:
            obj_logits_3d, rel_cls_3d, obj_logits_2d, object_label_backup = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(), descriptor, batch_ids, gt_cls, istrain=False)
        else:
            obj_logits_3d, rel_cls_3d, obj_logits_2d = torch.rand(obj_points.shape[0], self.num_class).cuda(self.config.DEVICE), torch.rand_like(gt_rel_cls).cuda(self.config.DEVICE), torch.rand(obj_points.shape[0], self.num_class).cuda(self.config.DEVICE)

        # compute metric

        if self.mconfig.use_obj_filter:
            top_k_obj_3d = evaluate_topk_object_w_backup(obj_logits_3d.detach(), gt_cls, topk=11, backup=object_label_backup)  # 
        else:
            top_k_obj_3d = evaluate_topk_object(obj_logits_3d.detach(), gt_cls, topk=11)  # 

        top_k_obj_2d = evaluate_topk_object(obj_logits_2d.detach(), gt_cls, topk=11)  # 
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)  # Convert from one-hot encoding to a list of categories.
        top_k_rel = evaluate_topk_predicate(rel_cls_3d.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        

        # result_print.cal_gt_and_predict(rel_cls_3d.detach().cpu(), gt_edges, obj_logits_3d.detach().cpu(), gt_cls.cpu(), scan_id, split_id, origin_obj_points, edge_indices.cpu(), batch_ids.cpu())
        # if (top_k_rel <= 1).sum() / len(top_k_rel) < 0.7:
        #     result_print.error_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
        # if (top_k_rel <= 1).sum() / len(top_k_rel) > 0.9:
        #     if (top_k_obj_3d <= 1).sum() / len(top_k_obj_3d) > 0.8:
        #         result_print.correct_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
        # result_print.all_scan_num += 1

        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_logits_3d.detach(), rel_cls_3d.detach(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=True, obj_topk=top_k_obj_3d)
            
            if cal_recall:
                triplet_recall_list_wo_gc = evaluate_triplet_recallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1000, use_clip=False, evaluate='triplet')
                relation_recall_list_wo_gc = evaluate_triplet_recallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1000, use_clip=False, evaluate='rels')
                
                triplet_recall_list_w_gc = evaluate_triplet_recallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1, use_clip=False, evaluate='triplet')
                relation_recall_list_w_gc = evaluate_triplet_recallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1, use_clip=False, evaluate='rels')

                triplet_mean_recall_list_w_gc = evaluate_triplet_mrecallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1, use_clip=False, evaluate='triplet')
                relation_mean_recall_list_w_gc = evaluate_triplet_mrecallk(obj_logits_3d.detach().cpu(), rel_cls_3d.detach().cpu(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, [20, 50, 100], 1, use_clip=False, evaluate='rels')
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        if cal_recall:
            return top_k_obj_3d, top_k_obj_2d, top_k_rel, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores, \
                triplet_recall_list_wo_gc, relation_recall_list_wo_gc, triplet_recall_list_w_gc, relation_recall_list_w_gc, \
                triplet_mean_recall_list_w_gc, relation_mean_recall_list_w_gc
        else:
            return top_k_obj_3d, top_k_obj_2d, top_k_rel, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
 
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x