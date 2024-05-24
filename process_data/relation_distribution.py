import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from utils.eva_utils_acc import get_gt
import math
from process_data.statics import Statics
import os

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/train_scans.txt'))
        with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/validation_scans.txt'))
        with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)
    
    return data

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    
def data_to_scene_for_excel(data):
    scene_data = dict()
    for scan in data['scans']:
        object_dict = scan["objects"]
        for rel in scan["relationships"]:
            try:
                sub = object_dict[str(rel[0])]
                obj = object_dict[str(rel[1])]
            except:
                print(scan["scan"])
                continue
            key = tuple([sub, obj])
            if key not in scene_data.keys():
                scene_data[key] = {rel[3] : 1}
            elif rel[3] not in scene_data[key].keys():
                scene_data[key][rel[3]] = 1
            else:
                scene_data[key][rel[3]] += 1

    return scene_data

# def create_scene_data_weight_seesaw():
#     data = read_json("train")
#     scene_data = data_to_scene_for_excel(data)
#     idx_to_obj = {}
#     relation_dict = {}
#     weight_dict = {}
#     L = 0
#     alpha = 0.9
#     with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/relationships.txt", "r") as file:
#         relation_list = file.read().splitlines()
#         relation_list.pop(0)
#         L = len(relation_list)
#         for idx, i in enumerate(relation_list):
#             relation_dict[i] = idx
    
#     with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/classes.txt", "r") as file:
#         obj_list = file.read().splitlines()
#         for idx, i in enumerate(obj_list):
#             idx_to_obj[idx] = i
    
#     for i, key in enumerate(tqdm(scene_data.keys())):
#         weight = torch.ones([L, L])
#         for item_i in scene_data[key].items():
#             for item_j in scene_data[key].items():
#                 i_num = item_i[1]
#                 j_num = item_j[1]
#                 i_idx = relation_dict[item_i[0]]  # 关系的标签索引按照relations里面的顺序
#                 j_idx = relation_dict[item_j[0]]
#                 if i_num > j_num:
#                     weight[i_idx, j_idx] = (j_num / i_num) ** alpha
        
#         weight_dict[key] = weight
#     return weight_dict, idx_to_obj


# def create_scene_data_weight_binary():
#     data = read_json("train")
#     scene_data = data_to_scene_for_excel(data)
#     idx_to_obj = {}
#     relation_dict = {}
#     weight_dict = {}
#     L = 0
#     alpha = 0.9
#     with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/relationships.txt", "r") as file:
#         relation_list = file.read().splitlines()
#         relation_list.pop(0)
#         L = len(relation_list)
#         for idx, i in enumerate(relation_list):
#             relation_dict[i] = idx
    
#     with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/classes.txt", "r") as file:
#         obj_list = file.read().splitlines()
#         for idx, i in enumerate(obj_list):
#             idx_to_obj[idx] = i
    
#     for i, key in enumerate(tqdm(scene_data.keys())):
#         weight = torch.ones([L])
#         for item_i in scene_data[key].items():
#             i_num = item_i[1]
#             i_idx = relation_dict[item_i[0]]
#             weight[i_idx] = abs(1 - 1.0 / (math.log(i_num + 1)+1))
#         weight_dict[key] = weight
#     return weight_dict, idx_to_obj


class Result_print():
    def __init__(self, exp_name, classNames, relationNames, use_rio27) -> None:
        self.root = "/data/lyf/3DSSG_code/WS_vlsat/process_data/result"
        self.relationNames = relationNames
        self.classNames = classNames
        self.triplet_result = {}
        self.thredshold = 0.5
        self.relation_dict, self.obj_dict = self.init_obj_and_rel_dict()
        self.sum_relation = {value:np.array([]) for key, value in self.relation_dict.items()}
        self.sum_relation["None"] = np.array([])
        self.sum_object = {value:np.array([]) for key, value in self.obj_dict.items()}
        self.scene_data = []
        self.error_scene = []
        self.correct_scene = []
        self.all_scan_num = 0
        self.exp_name = exp_name
        if not os.path.exists(os.path.join(self.root, self.exp_name)):
            os.mkdir(os.path.join(self.root, self.exp_name))
        self.static = Statics(classNames, relationNames, use_rio27)
        self.static.count_train_val()
    
    def init_obj_and_rel_dict(self):
        relation_dict = {}
        obj_dict = {}
        L = len(self.relationNames)
        for idx, i in enumerate(self.relationNames):
            relation_dict[idx] = i
    
        for idx, i in enumerate(self.classNames):
            obj_dict[idx] = i

        return relation_dict, obj_dict

    def cal_gt_and_predict(self, rel_predict, gt_edges, objs_pred, objs_gt, scene_id, split_id, obj_points, edge_indices, batch_ids):
        # gt_edges = get_gt(gt_cls, gt_rel_cls, edge_indices, True)
        self.append_to_rel_result(rel_predict, gt_edges, 6)
        self.append_to_object_result(objs_pred, objs_gt)
        self.count_scene_result_dict(rel_predict, gt_edges, objs_pred, objs_gt, scene_id, split_id, obj_points, edge_indices, batch_ids)
        

        for rel in range(len(rel_predict)):
            rel_pred = rel_predict[rel]
            sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
            rels_target = gt_edges[rel][2]
            
            # if len(rels_target) == 0: # no gt relation
            #     if rel_pred[sorted_idx[0]] < self.thredshold:
            #         self.append_to_result(gt_edges[rel], "None", "None")
            #     else:
            #         self.append_to_result(gt_edges[rel], "None", int(sorted_idx[0]))
            
            for true_rel in rels_target:
                self.append_to_triplet_result(gt_edges[rel], true_rel, int(sorted_idx[0]))


    def count_scene_result_dict(self, rel_predict, gt_edges, objs_pred, objs_gt, scene_id, split_id, obj_points, edge_indices, batch_ids):
        split_batch_ids = [sum(batch_ids == i) for i in batch_ids.unique()]
        split_rel_ids_num = [i*(i-1) for i in split_batch_ids]

        split_rel_ids = []
        s = 0
        s_num = 0
        for i in split_batch_ids:
            t = i*(i-1)
            split_rel_ids.append(edge_indices[s:s+t, ...] - s_num)
            s += t
            s_num += i

        gt_edge_label = np.array([i[2] for i in gt_edges], dtype=object)

        objs_pred_list = objs_pred.split(split_batch_ids, dim=0)
        objs_gt_list = objs_gt.split(split_batch_ids, dim=0)
        scene_id_list = np.split(scene_id, split_batch_ids)
        split_id_list = np.split(split_id, split_batch_ids)

        rel_predict_list = rel_predict.split(split_rel_ids_num, dim=0)
        rel_gt_list = np.split(gt_edge_label, split_rel_ids_num)

        for s_obj_pred, s_obj_gt, s_scene_id, s_split_id, s_obj_points, s_rel_predict, s_rel_gt, s_edge_idx in zip(objs_pred_list, objs_gt_list, scene_id_list, split_id_list, obj_points, rel_predict_list, rel_gt_list, split_rel_ids):
            if scene_id_list:
                s_scene_id = np.unique(s_scene_id)
                s_split_id = np.unique(s_split_id)
                assert len(s_scene_id) == 1
                assert len(s_split_id) == 1

                self.scene_data.append({
                    "rel_predict" : s_rel_predict, 
                    "gt_edges" : s_rel_gt, 
                    "obj_pred" : s_obj_pred, 
                    "obj_gt" : s_obj_gt, 
                    "scene_id" : str(s_scene_id[0]), 
                    "scene_split" : str(s_split_id[0]),
                    "obj_points" : s_obj_points, 
                    "edge_indices" : s_edge_idx
                })


    
    def append_to_object_result(self, objs_pred, objs_target, topk=11):
        for obj in range(len(objs_pred)):
            obj_pred = objs_pred[obj]  # 取出一个物体在所有类别上的概率
            sorted_idx = torch.sort(obj_pred, descending=True)[1]  # 根据概率排序
            gt = int(objs_target[obj])
            index = 1
            for idx in sorted_idx:
                if obj_pred[gt] >= obj_pred[idx] or index > topk:
                    break
                index += 1
            self.sum_object[self.obj_dict[gt]] = np.concatenate((self.sum_object[self.obj_dict[gt]], np.array([index])))

    
    def append_to_rel_result(self, rels_preds, gt_edges, topk, confidence_threshold=0.5, epsilon=0.02):
        for rel in range(len(rels_preds)):
            rel_pred = rels_preds[rel]
            
            sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
            rels_target = gt_edges[rel][2]
            
            if len(rels_target) == 0: # no gt relation
                indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
                if len(indices) == 0:
                    index = topk + 1
                else:
                    index = sorted(indices)[0].item()+1
                self.sum_relation["None"] = np.concatenate((self.sum_relation["None"], np.array([index])))
            else:
                temp_topk = []
                for gt in rels_target:
                    index = 1
                    for idx in sorted_idx:
                        if rel_pred[gt] >= rel_pred[idx] or index > topk:
                            break
                        index += 1
                    temp_topk.append(index)
                topk_idx = np.argsort(temp_topk)
                counter = 0
                for idx in topk_idx:
                    tmp = temp_topk[idx]
                    rel_name = self.relation_dict[rels_target[idx]]
                    self.sum_relation[rel_name] = np.concatenate((self.sum_relation[rel_name], np.array([tmp - counter])))
                    counter += 1

        
    def append_to_triplet_result(self, key, true_rel, predict_rel):
        key = tuple([self.obj_dict[int(key[0].cpu())], self.obj_dict[int(key[1].cpu())]])
        
        if type(true_rel) == int:
            true_rel = self.relation_dict[true_rel]
        if type(predict_rel) == int:
            predict_rel = self.relation_dict[predict_rel]

        if key not in self.triplet_result.keys():
            self.triplet_result[key] = {true_rel : {predict_rel : 1}}
        elif true_rel not in self.triplet_result[key].keys():
            self.triplet_result[key][true_rel] = {predict_rel : 1}
        elif predict_rel not in self.triplet_result[key][true_rel].keys():
            self.triplet_result[key][true_rel][predict_rel] = 1
        else:
            self.triplet_result[key][true_rel][predict_rel] += 1

    def print_result(self, name):
        self.print_sum_result()
        self.save_scene_data()
        self.print_triplet_result(name)
        self.print_error_scan()
        self.print_rel_error_predict()
        Max = 0
        for v in self.triplet_result.values():
            if len(v.keys()) > Max:
                Max = len(v.keys())
        
        column = ["sub", "obj"]
        for i in range(Max):
            s = "true_rel_%d" % i
            n = "pre_rel_%d" % i
            column.append(s)
            column.append(n)
        
        df = pd.DataFrame(data=None, columns=column)

        for i, key in enumerate(tqdm(self.triplet_result.keys())):
            col = {"sub" : key[0], "obj" : key[1]}
            for idx, item in enumerate(self.triplet_result[key].items()):
                s = "true_rel_%d" % idx
                n = "pre_rel_%d" % idx
                col[s] = item[0]
                col[n] = list(item[1].items())
            col = pd.Series(col)
            df = pd.concat([df, col.to_frame().T], ignore_index=True)
        df.sort_values(by="sub" , inplace=True, ascending=True) 
        df.to_excel(os.path.join(self.root, self.exp_name, "triplet_predict.xlsx"), index=False)


    def print_rel_error_predict(self):
        temp_dict = {name : {} for name in self.relation_dict.values()}
        for values in self.triplet_result.values():
            for true_rel, predict_rel_dict in values.items():
                for pre_rel, num in predict_rel_dict.items():
                    if pre_rel in temp_dict[true_rel].keys():
                        temp_dict[true_rel][pre_rel] += num
                    else:
                        temp_dict[true_rel][pre_rel] = num
        
        column = ["relation", "error_predict", "prediction_num"]
        df = pd.DataFrame(data=None, columns=column)
        for true_rel, values in temp_dict.items():
            col = {"relation" : true_rel}
            num_true = sum([int(i) for i in values.values()])
            error_predict = ""
            index = np.argsort(np.array(list(values.values())))
            value_item = list(values.items())
            for idx in index:
                pred_rel, num = value_item[idx]
                if pred_rel == true_rel:
                    continue
                t = "%s: %d(%.2f)  " % (pred_rel, num, num / num_true * 100)
                error_predict += t
            col["error_predict"] = error_predict
            col["prediction_num"] = num_true
            col = pd.Series(col)
            df = pd.concat([df, col.to_frame().T], ignore_index=True)
        df.to_excel(os.path.join(self.root, self.exp_name, "error_predict_result.xlsx"), index=False)



    def print_triplet_result(self, name):
        column = ["sub", "obj", "correct_num", "all_num", "ratio", "true_relations"]
        df = pd.DataFrame(data=None, columns=column)
        for i, key in enumerate(self.triplet_result.keys()):
            col = {"sub" : key[0], "obj" : key[1]}
            correct = 0.0
            all = 0.0
            for k, v in self.triplet_result[key].items():
                if k in v.keys():
                    correct += v[k]
                else:
                    correct += 0
                all += sum(list(v.values()))
            col["correct_num"] = correct
            col["ratio"] = float(format(correct / all * 100,'.2f'))
            col["all_num"] = all
            col["true_relations"] = ','.join(list(self.triplet_result[key].keys()))
            col = pd.Series(col)
            df = pd.concat([df, col.to_frame().T], ignore_index=True)
        df.sort_values(by="ratio" , inplace=True, ascending=True)
        df.to_excel(os.path.join(self.root, self.exp_name, "triplet_summerize.xlsx"), index=False)

    def print_error_scan(self):
        print("---------------------------------------------------------------------")
        print("The scene that ratio of rel error more than 30%")
        print("error scene: %d   correct scene: %d  all_scene: %d" % 
              (len(self.error_scene), self.all_scan_num - len(self.error_scene), self.all_scan_num))
        for i in self.error_scene:
            print("%s_%s  correct ratio:%.2f  total_rel:%d" % (i[0], i[1], i[2], i[3]))
        print("---------------------------------------------------------------------")

        print("The scene that ratio of rel correct more than 90%")
        print("correct scene: %d   error scene: %d  all_scene: %d" % 
              (len(self.correct_scene), self.all_scan_num - len(self.correct_scene), self.all_scan_num))
        for i in self.correct_scene:
            print("%s_%s  correct ratio:%.2f  total_rel:%d" % (i[0], i[1], i[2], i[3]))
        print("---------------------------------------------------------------------")

    def print_sum_result(self):
        """统计每一种关系和物体的预测结果"""
        column = ["relation", "top1", "top3", "top5", "length", "train_num", "val_num", "index"]
        df_rel = pd.DataFrame(data=None, columns=column)
        train_all = sum(list(self.static.count_train_rel.values()))
        val_all = sum(list(self.static.count_val_rel.values()))
        for key, value in self.sum_relation.items():
            L = len(value)
            col = {"relation" : key}
            col["top1"] = "%d(%.2f)" % ((value <= 1).sum(), (value <= 1).sum() / L * 100)
            col["top3"] = "%d(%.2f)" % ((value <= 3).sum(), (value <= 3).sum() / L * 100)
            col["top5"] = "%d(%.2f)" % ((value <= 5).sum(), (value <= 5).sum() / L * 100)
            col["length"] = L
            col["train_num"] = "%d(%.2f)" % (self.static.count_train_rel[key], self.static.count_train_rel[key] / train_all * 100)
            col["val_num"] = "%d(%.2f)" % (self.static.count_val_rel[key], self.static.count_val_rel[key] / val_all * 100)
            col["index"] = (value <= 1).sum() / L
            col = pd.Series(col)
            df_rel = pd.concat([df_rel, col.to_frame().T], ignore_index=True)
        df_rel.sort_values(by="index" , inplace=True, ascending=True)
        df_rel = df_rel.drop("index", axis=1)
        df_rel.to_excel(os.path.join(self.root, self.exp_name, "rel_top_rank.xlsx"), index=False)
        

        column = ["object", "top1", "top5", "top10", "length", "train_num", "val_num", "index"]
        df_obj = pd.DataFrame(data=None, columns=column)
        train_all = sum(list(self.static.count_train_obj.values()))
        val_all = sum(list(self.static.count_val_obj.values()))
        for key, value in self.sum_object.items():
            L = len(value)
            col = {"object" : key}
            col["top1"] = "%d(%.2f)" % ((value <= 1).sum(), (value <= 1).sum() / L * 100)
            col["top5"] = "%d(%.2f)" % ((value <= 5).sum(), (value <= 5).sum() / L * 100)
            col["top10"] = "%d(%.2f)" % ((value <= 10).sum(), (value <= 10).sum() / L * 100)
            col["length"] = len(value)
            col["train_num"] = "%d(%.2f)" % (self.static.count_train_obj[key], self.static.count_train_obj[key] / train_all * 100)
            col["val_num"] = "%d(%.2f)" % (self.static.count_val_obj[key], self.static.count_val_obj[key] / val_all * 100)
            col["index"] = (value <= 1).sum() / L
            col = pd.Series(col)
            df_obj = pd.concat([df_obj, col.to_frame().T], ignore_index=True)
        df_obj.sort_values(by="top1" , inplace=True, ascending=True)
        df_obj = df_obj.drop("index", axis=1)
        df_obj.to_excel(os.path.join(self.root, self.exp_name, "obj_top_rank.xlsx"), index=False)


    def save_scene_data(self):
        np.save(os.path.join(self.root, self.exp_name, "scene_result.npy"), self.scene_data)





def dict_to_pd(scene_data):
    Max = 0
    for v in scene_data.values():
        if len(v.keys()) > Max:
            Max = len(v.keys())
    
    column = ["sub", "obj"]
    for i in range(Max):
        s = "relation_%d" % i
        n = "num_%d" % i
        column.append(s)
        column.append(n)
    
    df = pd.DataFrame(data=None, columns=column)

    for i, key in enumerate(scene_data.keys()):
        col = {"sub" : key[0], "obj" : key[1]}
        for idx, item in enumerate(scene_data[key].items()):
            s = "relation_%d" % idx
            n = "num_%d" % idx
            col[s] = item[0]
            col[n] = item[1]
        col = pd.Series(col)
        df = pd.concat([df, col.to_frame().T], ignore_index=True)
    df.sort_values(by="sub" , inplace=True, ascending=True) 
    df.to_excel("result.xlsx", index=False)


if __name__ == "__main__":
    scene_data = read_json("train")
    dict_to_pd(scene_data)