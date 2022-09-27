from pyquaternion import Quaternion
import numpy as np
import random

def get_scene_lidar_token(nusc, scene_token, frame_skip=2):
    sensor = 'LIDAR_TOP'
    scene = nusc.get('scene', scene_token)
    first_sample = nusc.get('sample', scene['first_sample_token'])
    lidar = nusc.get('sample_data', first_sample['data'][sensor])

    lidar_token_list = [lidar['token']]
    counter = 1
    while lidar['next'] != '':
        lidar = nusc.get('sample_data', lidar['next'])
        counter += 1
        if counter % frame_skip == 0:
            lidar_token_list.append(lidar['token'])
    return lidar_token_list


def get_lidar_token_list(nusc, frame_skip, mode):
    scene_list = []
    for scene in nusc.scene:
        scene_list.append(scene['token'])
    if mode == 'train': scene_list = scene_list[:700]
    elif mode == 'valid': scene_list = scene_list[700:]
    print(mode + ' scenes length: ', len(scene_list))

    lidar_token_list = []
    for scene_token in scene_list:
        lidar_token_list += get_scene_lidar_token(nusc, scene_token, frame_skip=frame_skip)
    return lidar_token_list

def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P

def get_calibration_P(nusc, sample_data):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
    t = np.asarray(calib['translation']).astype(np.float32)
    P = get_P_from_Rt(R, t)
    return P

def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P

def get_camera_K(nusc, camera):
    calib = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    return np.asarray(calib['camera_intrinsic']).astype(np.float32)

def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]

def search_nearby_cameras(nusc,
                          init_camera,
                          max_translation,
                          direction,
                          lidar_P_inv,
                          nearby_camera_token_list):
    init_camera_direction_token = init_camera[direction]
    if init_camera_direction_token == '':
        return nearby_camera_token_list

    camera = nusc.get('sample_data', init_camera_direction_token)
    while True:
        camera_token = camera[direction]
        if camera_token == '':
            break
        camera = nusc.get('sample_data', camera_token)
        camera_P = get_sample_data_ego_pose_P(nusc, camera)
        P_lc = np.dot(lidar_P_inv, camera_P)
        t_lc = P_lc[0:3, 3]
        t_lc_norm = np.linalg.norm(t_lc)

        if t_lc_norm < max_translation:
            nearby_camera_token_list.append(camera_token)
        else:
            break
    return nearby_camera_token_list


def get_nearby_camera_token_list(nusc,
                                 lidar_token,
                                 max_translation,
                                 camera_name):
    lidar = nusc.get('sample_data', lidar_token)
    lidar_P = get_sample_data_ego_pose_P(nusc, lidar)
    lidar_P_inv = np.linalg.inv(lidar_P)

    lidar_sample_token = lidar['sample_token']
    lidar_sample = nusc.get('sample', lidar_sample_token)

    init_camera_token = lidar_sample['data'][camera_name]
    init_camera = nusc.get('sample_data', init_camera_token)
    nearby_camera_token_list = [init_camera_token]

    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'next',
        lidar_P_inv,
        nearby_camera_token_list)
    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'prev',
        lidar_P_inv,
        nearby_camera_token_list)

    return nearby_camera_token_list


def get_nearby_camera(nusc, lidar_token, max_translation):
    cam_list = ['CAM_FRONT',
                # 'CAM_FRONT_LEFT',
                # 'CAM_FRONT_RIGHT',
                # 'CAM_BACK',
                # 'CAM_BACK_LEFT',
                # 'CAM_BACK_RIGHT'
                ]
    nearby_cam_token_dict = {}
    for camera_name in cam_list:
        nearby_cam_token_dict[camera_name] \
            = get_nearby_camera_token_list(nusc,
                                           lidar_token,
                                           max_translation,
                                           camera_name)
    return nearby_cam_token_dict


def make_nuscenes_dataset(nusc, frame_skip, max_translation, mode):
    dataset = []

    lidar_token_list = get_lidar_token_list(nusc, frame_skip, mode)
    for i, lidar_token in enumerate(lidar_token_list):
        # begin_t = time.time()
        nearby_camera_token_dict = get_nearby_camera(nusc, lidar_token, max_translation)
        nearby_camera_tokens = nearby_camera_token_dict['CAM_FRONT']
        nearby_camera_token = random.choice(nearby_camera_tokens)

        dataset.append((lidar_token, nearby_camera_token))
        # if i == 10 : break

        # print('lidar %s takes %f' % (lidar_token, time.time()-begin_t))
        if i % 1000 == 0:
            print('%d done...' % i)

    return dataset
