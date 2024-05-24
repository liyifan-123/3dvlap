from asyncio import sleep
import json, os, trimesh, argparse
from operator import index
from xml.dom import INDEX_SIZE_ERR
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import clip, torch
import numpy as np
import time

DEVICE = "cuda:1"
model, preprocess = clip.load("ViT-B/32", device=DEVICE)


def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def process_imgs(imgs):
    # rotate images
    a = torch.stack([preprocess(Image.fromarray(img).transpose(Image.ROTATE_270)).cuda(DEVICE)for img in imgs], dim=0)
    return a

def read_pointcloud(scan_id):
    """
    Reads a pointcloud from a file and returns points with instance label.
    """
    plydata = trimesh.load(os.path.join('/data/lyf/3DSSG_code/3dvlap/data/3RScan', scan_id, 'labels.instances.align.annotated.v2.ply'), process=False)
    points = np.array(plydata.vertices)  
    labels = np.array(plydata.metadata['ply_raw']['vertex']['data']['objectId']) 

    return points, labels

def get_label(label_path):
    label_list = []
    with open(label_path, "r") as f:
        data = f.readlines()
    for line in data:
        label_list.append(line.strip())
    # get norm clip weight

    return label_list
    
def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/train_scans.txt'))
        with open("/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/validation_scans.txt'))
        with open("/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    # convert data to dict('scene_id': {'obj': [], 'rel': []})
    scene_data = dict()
    for i in data['scans']:
        if i['scan'] not in scene_data.keys():
            scene_data[i['scan']] = {'obj': dict(), 'rel': list()}
        scene_data[i['scan']]['obj'].update(i['objects'])
        scene_data[i['scan']]['rel'].extend(i['relationships'])

    return scene_data, selected_scans

def read_intrinsic(intrinsic_path, mode='rgb'):
    with open(intrinsic_path, "r") as f:
        data = f.readlines()
    
    m_versionNumber = data[0].strip().split(' ')[-1]
    m_sensorName = data[1].strip().split(' ')[-2]
    
    if mode == 'rgb':
        m_Width = int(data[2].strip().split(' ')[-1])  
        m_Height = int(data[3].strip().split(' ')[-1])  
        m_Shift = None
        m_intrinsic = np.array([float(x) for x in data[7].strip().split(' ')[2:]])
        m_intrinsic = m_intrinsic.reshape((4, 4))
    else:
        m_Width = int(data[4].strip().split(' ')[-1])
        m_Height = int(data[5].strip().split(' ')[-1])
        m_Shift = int(data[6].strip().split(' ')[-1])
        m_intrinsic = np.array([float(x) for x in data[9].strip().split(' ')[2:]])  
        m_intrinsic = m_intrinsic.reshape((4, 4))
    
    m_frames_size = int(data[11].strip().split(' ')[-1])
    
    return dict(
        m_versionNumber=m_versionNumber,
        m_sensorName=m_sensorName,
        m_Width=m_Width,
        m_Height=m_Height,
        m_Shift=m_Shift,
        m_intrinsic=m_intrinsic,
        m_frames_size=m_frames_size
    )

def read_extrinsic(extrinsic_path):
    pose = []
    with open(extrinsic_path) as f:
        lines = f.readlines()
    for line in lines:
        pose.append([float(i) for i in line.strip().split()])
    return pose

def read_scan_info(scan_id, mode='rgb'):
    import cv2 as cv
    scan_path = os.path.join("/data/lyf/3DSSG_code/3dvlap/data/3RScan", scan_id)
    sequence_path = os.path.join(scan_path, "sequence")
    intrinsic_path = os.path.join(sequence_path, "_info.txt")
    intrinsic_info = read_intrinsic(intrinsic_path, mode='rgb')
    depth_intrinsic_info = read_intrinsic(intrinsic_path, mode='depth') 
    mode_template = 'color.jpg' if mode == 'rgb' else 'depth.pgm'
    
    image_list, extrinsic_list, depth_list = [], [], []
    
    for i in range(0, intrinsic_info['m_frames_size']):
        frame_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ mode_template)
        extrinsic_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "align.pose.txt")  
        depth_path = os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "depth.pgm") 
        assert os.path.exists(frame_path) and os.path.exists(extrinsic_path) and os.path.exists(depth_path)
        
        img = np.array(plt.imread(frame_path))
        image_list.append(img)

        # read rendered depth map
        # depth_img = plt.imread(depth_path) * 100
        # depth_img = Image.fromarray(depth_img).transpose(Image.ROTATE_90)
        # plt.imsave(os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "depth_new.jpg"), depth_img)

        with Image.open(depth_path) as d_img:
            depth_img = np.array(d_img.getdata())
            depth_img = depth_img.reshape(depth_intrinsic_info["m_Height"], depth_intrinsic_info["m_Width"])
            depth_img = Image.fromarray(depth_img.astype('int32')).resize((intrinsic_info["m_Width"], intrinsic_info["m_Height"]), Image.BILINEAR)
            depth_img = np.array(depth_img)
        # plt.imsave(os.path.join(sequence_path, "frame-%s." % str(i).zfill(6)+ "depth_new.jpg"), depth_img)
        

        depth_list.append(np.array(depth_img) * 0.001)
        # inverce the extrinsic matrix, from camera_2_world to world_2_camera
        extrinsic = np.matrix(read_extrinsic(extrinsic_path))
        extrinsic_list.append(extrinsic.I)
        # sleep(1)
    
    return np.array(image_list), np.array(extrinsic_list), intrinsic_info, depth_intrinsic_info, np.array(depth_list)


def map_pc_to_image(mp_dict):
    current_time = time.time()

    thread_max = 1
    scene_id = mp_dict["scene_id"]
    instance_names = mp_dict["instance_names"]
    save_path = mp_dict["save_path"]
    points, instances = read_pointcloud(scene_id)
    image_list, extrinsics, intrinsic_info, depth_intrinsic_info, depth_list = read_scan_info(scene_id)
    intrinsic, depth_intrinsic, width, height = intrinsic_info['m_intrinsic'], depth_intrinsic_info["m_intrinsic"], intrinsic_info['m_Width'], intrinsic_info['m_Height']

    points = torch.tensor(points).cuda(DEVICE)
    intrinsic = torch.tensor(intrinsic).cuda(DEVICE)
    extrinsics = torch.tensor(extrinsics).cuda(DEVICE)
    depth_list = torch.tensor(depth_list).cuda(DEVICE)

    instance_id = set(instance_names.keys()) 


    for i in tqdm(instance_id):
        # found the instance points, convert to homogeneous coordinates
        points_i = points[(instances==int(i)).flatten()]
        if points_i.shape[0] == 0:
            continue

        points_i = torch.hstack([points_i, torch.ones((points_i.shape[0],1)).cuda(DEVICE)])
        # transform to camera coordinates
        w_2_c = (extrinsics @ points_i.T)   # n_frames x 4 x n_points
        # transform to image coordinates
        c_2_i_origin = intrinsic[:3, :] @ w_2_c    # n_frames x 3 x n_points
        c_2_i_origin = c_2_i_origin.permute(0, 2, 1)    # n_frames x n_points x 3
        c_2_i = c_2_i_origin[...,:2] / c_2_i_origin[..., 2:] # n_frames x n_points x 2

        # find the points in the image
        N_img, N_points, _ = c_2_i.shape
        idx = 0
        quanlity = None
        croped_image_feats = []
        origin_image_feats = []
        thread = 0.1

        while quanlity is None and thread < thread_max:
            old_indexs = ((c_2_i[...,0]< width) & (c_2_i[...,0]>0) & (c_2_i[...,1]< height) & (c_2_i[...,1]>0))
            indexs = old_indexs.clone()

            for img in range(N_img):
                if True not in old_indexs[img]:
                    continue
                xy = c_2_i[img, old_indexs[img], :].long()
                depth_img = depth_list[img][xy[:, 1], xy[:, 0]]
                depth_point = c_2_i_origin[img, old_indexs[img], 2]
                indexs[img][old_indexs[img]] = torch.abs(depth_img - depth_point) < thread
            thread += 0.1


            float_indexs = indexs.to(torch.float16).mean(-1)
            # quanlity filter : quanlity B
            topk_index = torch.argsort(-indexs.to(torch.long).sum(-1)) 
            for k in topk_index:
                c_2_i_k = c_2_i[k][indexs[k]]  # [num_points, 2]
                image_i = image_list[k.cpu()]  # [540, 950, 3]
                if len(c_2_i_k) == 0:
                    continue

                padding_x = min(height * 0.3, 20)
                padding_y = min(width * 0.3, 20)
                left_up_x = max(0, int(c_2_i_k[...,1].min()) - padding_x)
                left_up_y = max(0, int(c_2_i_k[...,0].min()) - padding_y)
                right_down_x = min(int(c_2_i_k[...,1].max()) + padding_x, height)
                right_down_y = min(int(c_2_i_k[...,0].max()) + padding_y, width)
                img_ratio = (right_down_x - left_up_x)*(right_down_y - left_up_y) / (width*height)
                
                croped_image_i = image_i[left_up_x:right_down_x, left_up_y:right_down_y]
                
                # c_2_i_k = c_2_i[quanju_idx][indexs[quanju_idx].reshape(-1)]  # [num_points, 2]
                # points_image_i = tmp_input_img_list[k].copy()
                # for p in c_2_i_k:
                #     for x in range(int(p[0]) - 5, int(p[0]) + 5):
                #         for y in range(int(p[1]) - 5, int(p[1]) + 5):
                #             try:
                #                 points_image_i[y, x, 0] = 255
                #                 points_image_i[y, x, 1] = 0
                #                 points_image_i[y, x, 2] = 0
                #             except:
                #                 continue

                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_croped_view{idx}_score_{float_indexs[k]:.4f}_ratio_{img_ratio:.4f}_B.jpg'), croped_image_i)
                plt.imsave(os.path.join(save_path, f'instance_{i}_class_{instance_names[i]}_view{idx}_{k}_B.jpg'), image_i)
                    
                    # get image clip feature
                with torch.no_grad():
                    croped_image_feats.append(model.encode_image(preprocess(Image.fromarray(croped_image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda(DEVICE)).cpu().numpy())
                    origin_image_feats.append(model.encode_image(preprocess(Image.fromarray(image_i).transpose(Image.ROTATE_270)).unsqueeze(0).cuda(DEVICE)).cpu().numpy())
                    
                idx += 1
                if quanlity is None:
                    quanlity  = 'B'
                if idx == top_k:
                    break
        
        if thread >= thread_max and len(croped_image_feats) == 0:
            with torch.no_grad():
                croped_image_feats = origin_image_feats = [model.encode_text(clip.tokenize("there is a small object").cuda(DEVICE)).cpu().numpy()]
            print("instance_id: %s" % (i))
            fin_all.write(f"{scene_id}: instance_id: %s" % (i))

        # print(f'Scene:{scene_id} Instance:{i} Label:{instance_names[i]} Quanlity:{quanlity}')

        # store multi-view feature
        croped_image_feats_mean = np.concatenate(croped_image_feats, axis=0).mean(axis=0, keepdims=True)
        origin_image_feats_mean = np.concatenate(origin_image_feats, axis=0).mean(axis=0, keepdims=True)
        
        np.save(os.path.join(save_path, f'instance_{i}_croped_view_mean.npy'), croped_image_feats_mean) 
        np.save(os.path.join(save_path, f'instance_{i}_origin_view_mean.npy'), origin_image_feats_mean)
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='val', help='train or val')
    args = argparser.parse_args()
    global class_list
    global top_k
    top_k = 5
    print("========= Deal with {} ========".format(args.mode))

    # train
    class_list = get_label('/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/classes.txt')  # class_weight[160, 512]
    scene_data, selected_scans = read_json(args.mode) 
    # record global quanlity
    fin_all = open(os.path.join(f'/data/lyf/3DSSG_code/3dvlap/data/3DSSG_subset/{args.mode}_all_quanlity_no_fea_match.txt'), 'a')
    mp_list = []
    for i in selected_scans:
        t_dict = {}
        t_dict["scene_id"] = i
        t_dict["instance_names"] = scene_data[i]['obj']
        t_dict["save_path"] = f'/data/lyf/3DSSG_code/WS_vlsat/data/3RScan/{i}/multi_view_no_fea_match_top{top_k}'
        os.makedirs(t_dict["save_path"], exist_ok=True)
        mp_list.append(t_dict)
    print("finished preprocess!")

    for i, mp_dict in enumerate(mp_list):
        scene_id = mp_dict["scene_id"]
        print(f"=== {scene_id}:{i}/{len(mp_list)} ===")
        map_pc_to_image(mp_dict)
    fin_all.close()