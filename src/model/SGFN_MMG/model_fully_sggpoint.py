import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model.model_utils.model_base import BaseModel
from utils import op_utils
from src.utils.eva_utils_acc import get_gt, evaluate_topk_object, evaluate_topk_predicate, evaluate_triplet_topk
from src.model.model_utils.network_GNN import GraphEdgeAttenNetworkLayers
from src.model.model_utils.network_PointNet import PointNetfeat, PointNetCls, PointNetRelCls, PointNetRelClsMulti
from src.model.model_utils.network_sgpn import GraphTripleConvNet
from itertools import product
from src.model.model_utils.network_edge_gcn import Multi_EdgeGCN, NodeMLP, EdgeMLP, edge_feats_initialization

class SGGpoint(BaseModel):
    """
    512 + 256 baseline
    """
    def __init__(self, config, num_obj_class, num_rel_class, dim_descriptor=11):
        super().__init__('SGFN', config)

        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_obj_class
        self.num_rel=num_rel_class
        self.flow = 'target_to_source'
        self.clip_feat_dim = self.config.MODEL.clip_feat_dim

        dim_point_feature = 512
        
        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid
        
        # Object Encoder
        self.obj_encoder = PointNetfeat(
            device = config.DEVICE,
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512 - 8)      
        
        # Relationship Encoder
        self.rel_encoder = PointNetfeat(
            device = config.DEVICE,
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        
        self.mmg = Multi_EdgeGCN(512, 1024, 1, 1, 4)


        self.obj_predictor = NodeMLP(512, num_obj_class)
        self.rel_predictor = EdgeMLP(1024, num_rel_class)

        #self.init_weight()
        
        self.optimizer = optim.AdamW([
            {'params':self.obj_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_encoder.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.mmg.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.obj_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.rel_predictor.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.max_iteration, last_epoch=-1)
        self.optimizer.zero_grad()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)

    
    def forward(self, obj_points, obj_2d_feats, edge_indices, descriptor=None, batch_ids=None, istrain=False):

        obj_feature = self.obj_encoder(obj_points)

        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=1)
        
        ''' Create edge feature '''
        # with torch.no_grad():
        #     edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)
        
        # rel_feature = self.rel_encoder(edge_feature)
        

        split_batch_ids = [sum(batch_ids == i) for i in batch_ids.unique()]
        L = 0
        obj_fea_list = []
        rel_fea_list = []
        for i in range(len(split_batch_ids)):
            idx = (batch_ids == i).squeeze(-1)
            obj_temp_fea = obj_feature[idx]
            
            edge_indices = list(product(list(range(split_batch_ids[i])), list(range(split_batch_ids[i]))))
            edge_indices = torch.tensor([i for i in edge_indices if i[0]!=i[1]]).cuda(self.config.DEVICE)
            edge_temp_fea = edge_feats_initialization(obj_temp_fea, edge_indices)
            
            
            # edge_temp_fea = rel_feature[L : L+len(edge_indices), ...]
            L = L + len(edge_indices)

            node_fea, edge_fea = self.mmg(obj_temp_fea, edge_temp_fea, edge_indices)
            
            obj_fea_list.append(node_fea)
            rel_fea_list.append(edge_fea)

        obj_fea_list = torch.vstack(obj_fea_list)
        rel_fea_list = torch.vstack(rel_fea_list)

        obj_logits = self.obj_predictor(obj_fea_list.unsqueeze(0)).squeeze(0)
        rel_cls = self.rel_predictor(rel_fea_list.unsqueeze(0)).squeeze(0)
            
        return obj_logits, rel_cls

    def process_train(self, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, edge_indices, batch_ids=None, with_log=False, ignore_none_rel=False, weights_obj=None, weights_rel=None):
        self.iteration +=1    
        
        obj_pred, rel_pred = self(obj_points, obj_2d_feats, edge_indices.t().contiguous(),descriptor, batch_ids, istrain=True)
        
        # compute loss for obj
        loss_obj = F.cross_entropy(obj_pred, gt_cls)

         # compute loss for rel
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls.sum(-1) ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]                
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[0] = 0 # assume none is the first relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            elif self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'NONE':
                weight = None
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_pred.device, requires_grad=False)
            else:
                loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight = weight)
        
        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        
        loss = lambda_o * loss_obj + lambda_r * loss_rel
        self.backward(loss)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        if not with_log:
            return top_k_obj, top_k_rel, loss_rel.detach(), loss_obj.detach(), loss.detach()

        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        
        
        log = [("train/rel_loss", loss_rel.detach().item()),
                ("train/obj_loss", loss_obj.detach().item()),
                ("train/loss", loss.detach().item()),
                ("train/Obj_R1", obj_topk_list[0]),
                ("train/Obj_R5", obj_topk_list[1]),
                ("train/Obj_R10", obj_topk_list[2]),
                ("train/Pred_R1", rel_topk_list[0]),
                ("train/Pred_R3", rel_topk_list[1]),
                ("train/Pred_R5", rel_topk_list[2]),
            ]
        return log
           
    def process_val(self, result_print, obj_points, obj_2d_feats, gt_cls, descriptor, gt_rel_cls, 
                    edge_indices, batch_ids, scan_id, split_id, origin_obj_points, cal_recall=False, use_triplet=False):
 
        obj_pred, rel_pred = self(obj_points, None, edge_indices.t().contiguous(), descriptor, batch_ids, istrain=False)
        
        # compute metric
        top_k_obj = evaluate_topk_object(obj_pred.detach(), gt_cls, topk=11)
        gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, self.mconfig.multi_rel_outputs)
        top_k_rel = evaluate_topk_predicate(rel_pred.detach(), gt_edges, self.mconfig.multi_rel_outputs, topk=6)
        
        
        result_print.cal_gt_and_predict(rel_pred.detach().cpu(), gt_edges, obj_pred.detach().cpu(), gt_cls.cpu(), scan_id, split_id, origin_obj_points, edge_indices.cpu(), batch_ids.cpu())
        if (top_k_rel <= 1).sum() / len(top_k_rel) < 0.7:
            result_print.error_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
        if (top_k_rel <= 1).sum() / len(top_k_rel) > 0.9:
            if (top_k_obj <= 1).sum() / len(top_k_obj) > 0.8:
                result_print.correct_scene.append((scan_id[0][0], split_id[0][0], (top_k_rel <= 1).sum() / len(top_k_rel) * 100, len(top_k_rel)))
        result_print.all_scan_num += 1

        
        if use_triplet:
            top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(obj_pred.detach(), rel_pred.detach(), gt_edges, edge_indices, self.mconfig.multi_rel_outputs, topk=101, use_clip=False, obj_topk=top_k_obj)
        else:
            top_k_triplet = [101]
            cls_matrix = None
            sub_scores = None
            obj_scores = None
            rel_scores = None

        return top_k_obj, top_k_obj, top_k_rel, top_k_rel, top_k_triplet, top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores
     
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
