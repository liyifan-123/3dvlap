from graphviz import Digraph
import torch
import numpy as np
import os

# node_color_list = ['aliceblue', 'antiquewhite', 'cornsilk3', 'lightpink', 'salmon', 'palegreen', 'khaki',
#                    'darkkhaki', 'orange']
node_color_list = ["green", "#ff756c"]


def init_obj_and_rel_dict():
    relation_dict = {}
    obj_dict = {}
    with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/relationships.txt", "r") as file:
        relation_list = file.read().splitlines()
        relation_list.pop(0)
        L = len(relation_list)
        for idx, i in enumerate(relation_list):
            relation_dict[idx] = i
    
    with open("/data/lyf/3DSSG_code/VLSAR_link/data/3DSSG_subset/classes.txt", "r") as file:
        obj_list = file.read().splitlines()
        for idx, i in enumerate(obj_list):
            obj_dict[idx] = i
    return relation_dict, obj_dict



def get_rel_pred_list(rel_pred, rel_gt, th = 0.5):
    rel_pred_list = []
    for idx, rel in enumerate(rel_pred):
        gt = rel_gt[idx]
        sort_idx = np.argsort(rel)[::-1]
        if gt == []:
            if rel[sort_idx[0]] <= 0.5:
                rel_pred_list.append([])
            else:
                rel_pred_list.append([sort_idx[0]])
        else:
            rel_pred_list.append([sort_idx[0]])
    return np.array(rel_pred_list, dtype=object)


def visualize(scene_dict, test_id, with_GT=True):
    dot = Digraph(comment='The Scene Graph')
    dot.attr(rankdir='TB')  # 该有向图的布局方向为从上到下(Top-Bottom)

    relation_dict, obj_dict = init_obj_and_rel_dict()

    scene_id = scene_dict["scene_id"]
    scene_split = scene_dict["scene_split"]
    obj_pred = np.array(scene_dict["obj_pred"])
    obj_gt = np.array(scene_dict["obj_gt"])
    rel_pred = np.array(scene_dict["rel_predict"])
    rel_gt = np.array(scene_dict["gt_edges"])
    edge_indices = scene_dict["edge_indices"]

    # nodes
    obj_pred_cls = np.argmax(obj_pred, axis=1)
    dot.attr('node', shape='oval', fontname='Sans', fontsize='16.0')
    for index in range(len(obj_pred)):
        pred = obj_pred_cls[index]
        gt = obj_gt[index]
        color = node_color_list[0] if gt == pred else node_color_list[1]
        dot.attr('node', fillcolor=color, style='filled')
        if with_GT:
            note = str(index) + "-" + obj_dict[pred] + '\n(GT:' + obj_dict[gt] + ')'
        else:
            note = obj_dict[pred]
        dot.node(str(index), note)
    # edges
    rel_pred = get_rel_pred_list(rel_pred, rel_gt)
    dot.attr('edge', fontname='Sans', fontsize='12.0', color='black', style='filled')
    for index in range(len(rel_gt)):
        s, o = edge_indices[index]
        p = rel_pred[index]
        gt_p = rel_gt[index]
        if p == gt_p:
            dot.attr('edge', color='green')
        elif not p and gt_p:
            dot.attr('edge', color='#ff756c')
        elif p[0] in gt_p:
            dot.attr('edge', color='green')
        else:
            dot.attr('edge', color='#ff756c')
        if gt_p == []:   # ignore ground truth predicate is 'None'
            continue
        if with_GT:
            if p==[]:
                dot.edge(str(int(s)), str(int(o)), "None" + '\n(GT:' + relation_dict[gt_p[0]] + ')')
            else:
                dot.edge(str(int(s)), str(int(o)), relation_dict[p[0]] + '\n(GT:' + ','.join([relation_dict[i] for i in gt_p]) + ')')
        else:
            if p==[]:
                dot.edge(str(int(s)), str(int(o)), "None")
            else:
                dot.edge(str(int(s)), str(int(o)), relation_dict[p[0]])

    # print(dot.source)
    dot.render(filename=os.path.join("/data/lyf/3DSSG_code/WS_vlsat/process_data/vis", '{}/{}/scene_graph_{}'.format(scene_id, test_id, scene_split)),format="jpg")



if __name__=="__main__":
    test_id_list = ["ws_test_30_fully_sgpn_3RScan160", "ws_test_01_w_edge_att", "ws_test_05_fully_imp"]
    for test_id in test_id_list:

        data_list_path = "/data/lyf/3DSSG_code/WS_vlsat/process_data/result/{}/scene_result.npy".format(test_id)
    
        data_list = np.load(data_list_path, allow_pickle=True)
        scene_id_list = ["d7dc987e-a34a-2794-85c8-f75389b27532"]
        t = [i["scene_id"] for i in data_list]
        for i in data_list:
            if i["scene_id"] in scene_id_list:
                print(i["scene_id"])
                visualize(i, test_id)