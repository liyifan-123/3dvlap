from pydoc import describe
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import numpy as np

class CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter
    
class CustomMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter


class CustomDataLoader(DataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        if worker_init_fn is None:
            worker_init_fn = self.init_fn
        super().__init__(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.config = config
        
    def init_fn(self, worker_id):
        np.random.seed(self.config.SEED + worker_id)
        
    def __iter__(self):
        if self.num_workers == 0:
            return CustomSingleProcessDataLoaderIter(self)
        else:
            return CustomMultiProcessingDataLoaderIter(self)

def collate_fn_obj(batch):
    # batch
    
    name_list, instance2mask_list, obj_point_list, obj_label_list = [], [], [], []
    for i in batch:
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_point_list.append(i[2])
        obj_label_list.append(i[4])
    return name_list, instance2mask_list, torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel(batch):
    # batch
    name_list, instance2mask_list, obj_label_list, rel_point_list, rel_label_list, edge_indices = [], [], [], [], [], []
    for i in batch:
        assert len(i) == 7
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_label_list.append(i[4])
        rel_point_list.append(i[3])
        rel_label_list.append(i[5])
        edge_indices.append(i[6])
    return name_list, instance2mask_list, torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_obj_new(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel_new(batch):
    # batch
    rel_point_list, rel_label_list = [], []
    for i in batch:
        rel_point_list.append(i[1])
        rel_label_list.append(i[3])
    return torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0)


def collate_fn_all(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices = []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[3])
        rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5])

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_all_des(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
        #rel_point_list.append(i[1])
        rel_label_list.append(i[3])
        edge_indices.append(i[4] + count)
        descriptor.append(i[5])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_all_2d(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats = [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_2d_feats.append(i[1])
        obj_label_list.append(i[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5] + count)
        descriptor.append(i[6])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_det(batch):
    assert len(batch) == 1
    scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask = [], [], [], [], []
    for i in range(len(batch)):
        scene_points.append(batch[i][0])
        obj_boxes.append(batch[i][1])
        obj_labels.append(batch[i][2])
        point_votes.append(batch[i][3])
        point_votes_mask.append(batch[i][4])
    
    scene_points = torch.stack(scene_points, dim=0)
    obj_boxes = torch.stack(obj_boxes, dim=0)
    obj_labels = torch.stack(obj_labels, dim=0)
    point_votes = torch.stack(point_votes, dim=0)
    point_votes_mask = torch.stack(point_votes_mask, dim=0)

    return scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask


def collate_fn_mmg(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats, origin_obj_point_list = [], [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    batch_ids, scan_id, split_id = [], [], []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        obj_label_list.append(b[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(b[4])
        edge_indices.append(b[5] + count)
        descriptor.append(b[6])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i)) 
        scan_id.append(np.full((b[0].shape[0], 1), b[7]))
        split_id.append(np.full((b[0].shape[0], 1), b[8]))
        origin_obj_point_list.append(b[9])


    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
         np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list


def collate_fn_ws(batch):
    # batch
    obj_point_list, obj_2d_feats, origin_obj_point_list = [], [], []
    obj_label_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    batch_ids, scan_id, split_id = [], [], []
    obj_texts, tri_texts = [], []
    img_pair_info, img_pair_idx = [], []
    
    count_obj = 0
    count_rel = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        obj_label_list.append(b[2])
        rel_label_list.append(b[3])
        edge_indices.append(b[4] + count_obj)
        descriptor.append(b[5])
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))
        scan_id.append(np.full((b[0].shape[0], 1), b[6]))
        split_id.append(np.full((b[0].shape[0], 1), b[7]))
        origin_obj_point_list.append(b[8])
        obj_texts.append(b[9])
        tri_texts.append(b[10])
        if len(b[12]) != 0:
            img_pair_info.append(b[11])
            img_pair_idx.append(b[12] + count_rel)
        # accumulate batch number to make edge_indices match correct object index
        count_obj += b[0].shape[0]
        count_rel += len(b[4])

    
    if len(img_pair_info) != 0:
        return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
            torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
            np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list, obj_texts, tri_texts, torch.cat(img_pair_info, dim=0), torch.cat(img_pair_idx, dim=0)
    else:
        return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
            torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
            np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list, obj_texts, tri_texts, torch.tensor([]), torch.tensor([])



def collate_fn_trans(batch):
    # batch
    obj_point_list, obj_2d_feats, origin_obj_point_list, sampled_points_list, edge_pc_match_table_list = [], [], [], [], []
    obj_label_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    batch_ids, scan_id, split_id = [], [], []
    sub_obj_pair_text = []
    img_pair_info, img_pair_idx = [], []
    
    count_obj = 0
    count_rel = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        edge_pc_match_table_list.append(b[2])
        sampled_points_list.append(b[3])
        obj_label_list.append(b[4])
        rel_label_list.append(b[5])
        edge_indices.append(b[6] + count_obj)
        descriptor.append(b[7])
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))
        scan_id.append(np.full((b[0].shape[0], 1), b[8]))
        split_id.append(np.full((b[0].shape[0], 1), b[9]))
        origin_obj_point_list.append(b[10])
        sub_obj_pair_text.append(b[11])
        if len(b[13]) != 0:
            img_pair_info.append(b[12])
            img_pair_idx.append(b[13] + count_rel)
        # accumulate batch number to make edge_indices match correct object index
        count_obj += b[0].shape[0]
        count_rel += len(b[6])

    
    if len(img_pair_info) != 0:
        return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
            torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
            np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list, np.vstack(sub_obj_pair_text), \
                torch.cat(img_pair_info, dim=0), torch.cat(img_pair_idx, dim=0), torch.cat(edge_pc_match_table_list, dim=0), torch.vstack(sampled_points_list)
    else:
        return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
            torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
            np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list, np.vstack(sub_obj_pair_text), torch.tensor([]), torch.tensor([]), torch.cat(edge_pc_match_table_list, dim=0), torch.vstack(sampled_points_list)


def collate_fn_ws_few_shot(batch):
    # batch
    obj_point_list, obj_2d_feats, origin_obj_point_list = [], [], []
    obj_label_list, rel_label_list = [], []
    edge_indices, descriptor = [], []
    batch_ids, scan_id, split_id = [], [], []
    obj_texts, tri_texts, shot_idx = [], [], []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_2d_feats.append(b[1])
        obj_label_list.append(b[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(b[4])
        edge_indices.append(b[5] + count)
        descriptor.append(b[6])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        # Indicate which objects belong to the same scene
        batch_ids.append(torch.full((b[0].shape[0], 1), i)) 
        scan_id.append(np.full((b[0].shape[0], 1), b[7]))
        split_id.append(np.full((b[0].shape[0], 1), b[8]))
        origin_obj_point_list.append(b[9])
        obj_texts.append(b[10])
        tri_texts.append(b[11])
        shot_idx += b[12]


    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0),\
         np.vstack(scan_id), np.vstack(split_id), origin_obj_point_list, obj_texts, tri_texts, shot_idx



def collate_fn_scannet(batch):
    # batch
    obj_point_list = []
    edge_indices, descriptor = [], []
    batch_ids, scan_id_list, instance_list = [], [], []
    
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        edge_indices.append(b[1] + count)
        descriptor.append(b[2])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))
        scan_id_list.append(b[3])
        instance_list.append(b[4])


    return torch.cat(obj_point_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0), scan_id_list, instance_list