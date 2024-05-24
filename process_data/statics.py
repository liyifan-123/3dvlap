import json
import os
import pandas as pd
import numpy as np

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships 


class Statics():
    def __init__(self, classNames, relationNames, use_rio27) -> None:
        self.root = "/data/lyf/3DSSG_code/WS_vlsat/data/3DSSG_subset"
        self.data_train, self.data_val = self.dataset_loading_3RScan(use_rio27)
        self.relationNames = relationNames[:]
        self.classNames = classNames
        self.relationNames.insert(0, "None")
        self.count_train_rel = {name:0 for name in self.relationNames}
        self.count_train_obj = {name:0 for name in self.classNames}
        self.count_val_rel = {name:0 for name in self.relationNames}
        self.count_val_obj = {name:0 for name in self.classNames}
        self.same_edge_rel_train = {name:{} for name in self.relationNames}
        self.same_edge_rel_val = {name:{} for name in self.relationNames}

    def dataset_loading_3RScan(self, use_rio27): 
        if use_rio27:
            with open(os.path.join(self.root, 'relationships_rio27_train.json'), "r") as read_file:
                data_train = json.load(read_file)
            
            with open(os.path.join(self.root, 'relationships_rio27_validation.json'), "r") as read_file:
                data_val = json.load(read_file)
        else:
            with open(os.path.join(self.root, 'relationships_train.json'), "r") as read_file:
                data_train = json.load(read_file)
            
            with open(os.path.join(self.root, 'relationships_validation.json'), "r") as read_file:
                data_val = json.load(read_file)
        return  data_train, data_val
    
    def count_train_val(self):
        """计算训练集和验证集中不同类别关系的数量"""
        for scan in self.data_train["scans"]:
            for rel in scan["relationships"]:
                self.count_train_rel[rel[3]] += 1
            
            for obj in scan["objects"].values():
                self.count_train_obj[obj] += 1
        
        for scan in self.data_val["scans"]:
            for rel in scan["relationships"]:
                self.count_val_rel[rel[3]] += 1
            
            for obj in scan["objects"].values():
                self.count_val_obj[obj] += 1
    
    def print_train_val(self):
        train_all = sum(list(self.count_train_rel.values()))
        val_all = sum(list(self.count_val_rel.values()))
        print("relation数量对比:")
        for rel in self.relationNames:
            print("%s  train: %d(%.2f)  val: %d(%.2f)" % (rel, self.count_train_rel[rel], self.count_train_rel[rel] / train_all * 100, 
                                                          self.count_val_rel[rel], self.count_val_rel[rel] / val_all * 100))
        
        train_all = sum(list(self.count_train_obj.values()))
        val_all = sum(list(self.count_val_obj.values()))
        print("object数量对比:")
        for obj in self.classNames:
            print("%s  train: %d(%.2f)  val: %d(%.2f)" % (obj, self.count_train_obj[obj], self.count_train_obj[obj] / train_all * 100, 
                                                          self.count_val_obj[obj], self.count_val_obj[obj] / val_all * 100))
    
    def count_rel_in_same_edge(self):
        for scan in self.data_train["scans"]:
            temp_dict = {(rel[0], rel[1]):[] for rel in scan["relationships"]}
            for rel in scan["relationships"]:
                temp_dict[(rel[0], rel[1])].append(rel[3])
            
            for value in temp_dict.values():
                if len(value)>1:
                    for i in range(len(value)):
                        for j in range(len(value)):
                            if i==j:
                                continue
                            if value[j] not in self.same_edge_rel_train[value[i]].keys():
                                self.same_edge_rel_train[value[i]][value[j]] = 1
                            else:
                                self.same_edge_rel_train[value[i]][value[j]] += 1
        ### val
        for scan in self.data_val["scans"]:
            temp_dict = {(rel[0], rel[1]):[] for rel in scan["relationships"]}
            for rel in scan["relationships"]:
                temp_dict[(rel[0], rel[1])].append(rel[3])
            
            for value in temp_dict.values():
                if len(value)>1:
                    for i in range(len(value)):
                        for j in range(len(value)):
                            if i==j:
                                continue
                            if value[j] not in self.same_edge_rel_val[value[i]].keys():
                                self.same_edge_rel_val[value[i]][value[j]] = 1
                            else:
                                self.same_edge_rel_val[value[i]][value[j]] += 1
    
    def print_rel_in_same_edge(self):
        """统计哪些标签会出现在同一个边上"""
        column = ["relation", "same_edge"]
        df_train = pd.DataFrame(data=None, columns=column)
        for rel, values in self.same_edge_rel_train.items():
            value = " , ".join(["%s: %d" % (k, v) for k, v in values.items()])
            col = {"relation": rel, "same_edge": value}
            df_train = df_train.append(col, ignore_index=True)
        df_train.to_excel("relation_in_same_edge_train.xlsx", index=False)

        df_val = pd.DataFrame(data=None, columns=column)
        for rel, values in self.same_edge_rel_val.items():
            value = " , ".join(["%s: %d" % (k, v) for k, v in values.items()])
            col = {"relation": rel, "same_edge": value}
            df_val = df_val.append(col, ignore_index=True)
        df_val.to_excel("relation_in_same_edge_val.xlsx", index=False)
    

    def delete_few_shot_idx_file(self, folder):
        folder_list = os.listdir(folder)
        for f in folder_list:
            if os.path.exists(os.path.join(folder, f, "few_shot_idx.npy")):
                os.remove(os.path.join(folder, f, "few_shot_idx.npy"))
            if os.path.exists(os.path.join(folder, f, "few_shot_idx")):
                os.remove(os.path.join(folder, f, "few_shot_idx"))

    def save_few_shot_idx_file(self, file_name, split, idx):
        if os.path.exists(file_name):
            L = np.load(file_name).tolist()
            L[0].append(split)
            L[1].append(idx)
        else:
            L = [[split], [idx]]
            np.save(file_name, L)

    def choose_K_shot(self):
        """获取K-shot训练集"""
        folder = "/data/lyf/3DSSG_code/WS_vlsat/data/3RScan"
        self.delete_few_shot_idx_file(folder)
        K = 16
        result_dict = {i:[[], []] for i in self.relationNames}  # 分别对应scene_split和idx
        for scene in self.data_train["scans"]:
            object_dict = scene["objects"]
            scan = scene["scan"]
            split = scene["split"]
            key = scan + "_" + str(split)
            relation = scene["relationships"]
            for i, rel in enumerate(relation):
                result_dict[rel[3]][0].append(key)
                result_dict[rel[3]][1].append(i)
        
        rel_list = ["bigger than", "smaller than", "lower than", "higher than", "same as", "same symmetry as"]
        for rel_name, rel_res in result_dict.items():
            print("rel_name: ", rel_name)
            if rel_name not in rel_list:
                continue
            num_sample = len(rel_res[0])
            if num_sample == 0:
                continue
            indexs = np.random.choice(range(0,num_sample), K)
            file_idx = np.array(rel_res[0])[indexs]
            rel_idx = np.array(rel_res[1])[indexs]
            for i in range(len(file_idx)):
                s = file_idx[i]
                scene, split = s.split("_")
                print("%s_%s : %s" % (scene, split, rel_idx[i]))
                self.save_few_shot_idx_file(os.path.join(folder, scene, "few_shot_idx.npy"), split, rel_idx[i])

                










if __name__=="__main__":
    static = Statics()
    # static.count_train_val()
    # static.print_train_val()
    # static.count_rel_in_same_edge()
    # static.print_rel_in_same_edge()
    static.choose_K_shot()
            
