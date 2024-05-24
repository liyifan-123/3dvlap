import json
import os


def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output


def handle_obj(classNames, object_dict, object_rio27_dict):
    need_del_obj_id = []
    new_obj_dict = {}
    for key, value in object_dict.items():
        if object_rio27_dict[key]["rio27"] == "0":
            need_del_obj_id.append(key)
        else:
            new_obj_dict[key] = classNames[int(object_rio27_dict[key]["rio27"])]
    return new_obj_dict, need_del_obj_id


def handle_rel(relationNames, need_del_obj_id, relation_list):
    keep_relation = ['part of', 'left', 'cover', 'hanging in', 
                                 'belonging to', 'connected to', 'supported by', 
                                 'hanging on', 'right', 'attached to', 'build in', 
                                 'close by', 'behind', 'lying on', 'standing on', 
                                 'lying in', 'standing in', 'front', 'leaning against']
    new_relation_list = []
    for rel in relation_list:
        if str(rel[0]) in need_del_obj_id or str(rel[1]) in need_del_obj_id:
            continue
        if rel[-1] in keep_relation:
            rel[-1] = rel[-1].replace('left', 'spatial proximity')  \
                            .replace('right', 'spatial proximity') \
                            .replace('front', 'spatial proximity') \
                            .replace('behind', 'spatial proximity')
            rel[-2] = relationNames.index(rel[-1])
            new_relation_list.append(rel)
        else:
            continue
    return new_relation_list


def create_rio_27_datasets(classNames, relationNames, data):
    scene_ids_to_remove = ['a8952593-9035-254b-8f40-bc82e6bcbbb1',
                        '20c993b9-698f-29c5-87f1-4514b70070c3',
                        '20c99397-698f-29c5-8534-5304111c28af',
                        '20c993c7-698f-29c5-8685-0d1a2a4a3496',
                        'ae73fa15-5a60-2398-8646-dd46c46a9a3d',
                        '20c993c5-698f-29c5-8604-3248ede4091f',
                        '6bde60cd-9162-246f-8fad-fca80b4d6ad8',
                        '77941464-cfdf-29cb-87f4-0465d3b9ab00',
                        '0cac75af-8d6f-2d13-8f9e-ed3f62665aed',
                        '0cac768a-8d6f-2d13-8dd3-3cbb7d916641',
                        'ba6fda98-a4c1-2dca-8230-bce60f5a0f85',
                        'd7d40d48-7a5d-2b36-97ad-692c9b56b508',
                        'd7d40d46-7a5d-2b36-9734-659bccb1c202',
                        '352e9c48-69fb-27a7-8a35-3dbf699637c8',
                        'ba6fdaa0-a4c1-2dca-80a9-df196c04fcc8',
                        'd7d40d40-7a5d-2b36-977c-4e35fdd5f03a',
                        '0cac75e6-8d6f-2d13-8e4a-72b0fc5dc6c3',
                        '38770cab-86d7-27b8-85cd-d55727c4697b',
                        '0cac768c-8d6f-2d13-8cc8-7ace156fc3e7']

    with open("/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/objects.json", 'r') as f:
        OBJ = json.load(f)
        OBJ_D = {i["scan"]:i for i in OBJ["scans"]}
    
    new_data = {"scans":[]}
    scan_list = []
    for scene_data in data["scans"]:
        object_dict = scene_data["objects"]
        relation_list = scene_data["relationships"]
        scene_id = scene_data["scan"]
        scene_split = scene_data["split"]

        if scene_id in scene_ids_to_remove:
            continue

        object_rio27_dict = {i["id"]:i for i in OBJ_D[scene_id]["objects"]}
        new_obj_dict, need_del_obj_id = handle_obj(classNames, object_dict, object_rio27_dict)
        new_relation_list = handle_rel(relationNames, need_del_obj_id, relation_list)
        
        if len(new_obj_dict.keys()) == 0 or len(new_relation_list) == 0:
            print(f"{scene_id} : {scene_split}")
            continue

        scene_data["objects"] = new_obj_dict
        scene_data["relationships"] = new_relation_list
        new_data["scans"].append(scene_data)
        scan_list.append(scene_data["scan"])
    
    return new_data, set(scan_list)
        

def dataset_loading_3RScan(root:str, split:str):  
    # read object class
    classNames = ['_', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter', 
    'shelf', 'curtain', 'pillow', 'clothes', 'ceiling', 'fridge', 'tv', 'towel', 'plant', 'box', 'nightstand', 
    'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']
    # read relationship class
    relationNames = ['supported by', 'attached to', 'standing on', 'lying on', 'hanging on', 
                    'connected to', 'leaning against', 'part of', 'belonging to', 'build in',
                    'standing in', 'cover', 'lying in', 'hanging in', 'spatial proximity', 'close by']
    # read relationship json
    selected_scans=set()
    if split == 'train_scans' :
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(root, 'train_scans.txt')))
        with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(read_txt_to_list(os.path.join(root, 'validation_scans.txt')))
        with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)
    return  classNames, relationNames, data, selected_scans

if __name__ == "__main__":
    root = "/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset"
    split = "validation_scans"
    classNames, relationNames, data, selected_scans = dataset_loading_3RScan(root, split)
    data, scan_list = create_rio_27_datasets(classNames, relationNames, data)
    with open(f'/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/relationships_rio27_{split.split("_")[0]}.json','w') as file:
        file.write(json.dumps(data, indent=2, ensure_ascii=False))
    scan_list = list(scan_list)
    with open(f'/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/{split.split("_")[0]}_rio27_scans.txt','w') as file:
        file.writelines([i + "\n" for i in scan_list])