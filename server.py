import asyncio
from io import DEFAULT_BUFFER_SIZE
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import traceback
from pydantic import BaseModel
from typing import List
import numpy as np
import json
import datetime

# define global variables
cam1_table, cam2_table = {}, {}
MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE = -1
MAX_NUM_FEATURES = 50
MAX_NUM_FRAME_FROM_LAST_TIME_SEE = 20
L2_SIMILARITY_THRESHOLD = 1e6
COSINE_SIMILARITY_THRESHOLD = 1e6
MAX_CHECK_OBSOLETE_INTERVAL = MAX_NUM_FRAME_FROM_LAST_TIME_SEE * 2
cam1_to_cam2_counter = {}
cam1_to_cam2_final, cam2_to_cam1_final = {}, {}
global_ls_stay_time = {}
check_obsolete_counter_1, check_obsolete_counter_2 = 0, 0

# util functions
def show_table():
    # print(len(cam2_to_cam1_final.items()))
    cam1_to_cam2 = None
    if len(cam2_to_cam1_final.items()) > 0:
        cam1_to_cam2 = {v:k for k, v in cam2_to_cam1_final.items()}
    # print('cam1 to cam 2: ', cam1_to_cam2)
    print('Cam1_ID\t\tFirst_Time_See\t\tNum_Features\t\tCam2_ID\t\tStay_Time')
    for id, data in cam1_table.items():
        if cam1_to_cam2 is not None and id in cam1_to_cam2.keys():
            # if cam1_to_cam2[id] in global_ls_stay_time.keys():
                # print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), cam1_to_cam2[id]), global_ls_stay_time[cam1_to_cam2[id]])
            # else:
            print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), cam1_to_cam2[id], ''))

        else:
            print('\n{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(id, data['first_time_see'], len(data['features']), '', ''))
    
    print('---------------------------------------------------')

        

def l2_similarity(f1, f2):
    if isinstance(f1, list):
        f1 = np.array(f1)
    if isinstance(f2, list):
        f2 = np.array(f2)
    return np.sum((f1 - f2) ** 2)

def cosine_similarity(f1, f2):
    if isinstance(f1, list):
        f1 = np.array(f1)
    if isinstance(f2, list):
        f2 = np.array(f2)
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def check_l2_similarity(input_feature: list, table: dict, threshold=L2_SIMILARITY_THRESHOLD) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ??? b???ng table
    min_distance = 1e9
    min_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        for feature in data['features']:
            distance = l2_similarity(input_feature, feature)
            if distance < min_distance:
                min_distance =  distance
                min_id = id
    
    # n???u kho???ng c??ch g???n nh???t nh??? h??n ng?????ng cho ph??p => 2 ?????i t?????ng n??y l?? 1
    if min_distance < threshold:
        return min_id
    else:
        return -1


def check_cosine_similarity(input_feature: list, table: dict) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ??? b???ng table
    max_similarity = -1
    max_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        for feature in data['features']:
            similarity = cosine_similarity(input_feature, feature)
            if similarity > max_similarity:
                max_similarity =  similarity
                max_id = id
    
    # print('max similiarity: ', max_similarity)
    return max_id


def check_cosine_similarity_2(input_feature: list, table: dict) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ??? b???ng table
    max_similarity = -1
    max_id = -1
    for id, data in table.items():
        if id == -1:
            continue
        sum_similarity = 0
        for feature in data['features']:
            similarity = cosine_similarity(input_feature, feature)
            sum_similarity += similarity
        mean_similarity = sum_similarity / len(data['features'])
        if mean_similarity > max_similarity:
            max_similarity =  mean_similarity
            max_id = id
    
    # print('max similiarity: ', max_similarity)
    return max_id



def update_cam2_to_cam1_final():
    for pair, count in cam1_to_cam2_counter.items():
        pass



def time2datetime(time):
    return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

class Item(BaseModel):
    time: float
    feature: list
    id: int
    
# define received data format
class ItemList(BaseModel):
    __root__: List[Item]


# define app
app = FastAPI()


# root path
@app.get('/')
async def root():
    return {'message': 'Hello World'}


# endpoint to receive data from camera 1
@app.post('/cam1')
async def cam1(item_list: ItemList):
    # duyet qua tat ca ca item nhan duoc
    # neu khong co doi tuong nao trong khung hinhf => cam1 gui len 1 list rong => se ko lam gi ca
    for item in item_list.__root__: 
        if item.id in cam1_table.keys():    # neu item id da co trong table
            myobject = cam1_table[item.id]  # get object from table

            # update object with condition
            if myobject['num_frame_from_last_time_update'] >= MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE and len(myobject['features']) < MAX_NUM_FEATURES:
                myobject['features'].append(item.feature)
                myobject['num_frame_from_last_time_update'] = -1
            
            # if not update
            myobject['last_time_see'] = item.time   # assign last_time_see (thoi gian gan nhat van nhin thay doi tuong)
            myobject['num_frame_from_last_time_see'] = 0
            myobject['num_frame_from_last_time_update'] += 1
        
        elif item.id != -1:   # neu item id chua co trong table
            # them doi tuong vao bang
            cam1_table[item.id] = {
                'first_time_see': item.time,
                'features': [item.feature],
                'last_time_see': item.time,
                'num_frame_from_last_time_see': 0,
                'num_frame_from_last_time_update': 0
            }
            # print(f'ID {item.id} entered cam 1 at {time2datetime(item.time)}')

    # # save cam1_table to json
    # with open('cam1_mbnv2_128x64_trt.json', 'w') as f:
    #     json.dump(cam1_table, f)

    #     # check feature hien tai cua item nay co giong voi feature nao trong bang cam 2 khong (c?? th??? c??c l???n kh??c ch??a gi???ng nh??ng l???n n??y l???i gi???ng)
    #     min_id = check_feature_similarity(item.feature, cam2_table, metric='l2', threshold=L2_SIMILARITY_THRESHOLD)
    #     if min_id != -1:  # n???u c??
    #         # n???u id n??y ch??a xu???t hi???n trong cam1_to_cam2_final (l???n ?????u ti??n c?? c??i gi???ng n?? ??? cam 2), th??m n?? v??o
    #         if item.id not in cam1_to_cam2_final:
    #             cam1_to_cam2_final[item.id] = min_id

    #         # n???u c???p (min_id, id) n??y ???? c?? t??? tr?????c ????, t??ng ????? c???ng c??? 2 c??i n??y l?? m???t l??n 1
    #         if (item.id, min_id) in cam1_to_cam2_counter:
    #             cam1_to_cam2_counter[(item.id, min_id)] += 1 # (cam1_id, cam2_id)
    #             # c???p nh???t id trong cam 2 gi???ng n?? nh???t
    #             if cam1_to_cam2_counter[(item.id, min_id)] > cam1_to_cam2_counter[(item.id, cam1_to_cam2_final[item.id])]:
    #                 cam1_to_cam2_final[item.id] = min_id
    #         else: # n???u c???p (min_id, id) n??y ch??a c?? => th??m v??o 
    #             cam1_to_cam2_counter[(item.id, min_id)] = 1

    # # duyet qua tat ca nhung id khong xuat hien o lan nhan nay trong bang cam 1
    # ls_id_received = [item.id for item in item_list.__root__]
    # ls_delete = []
    # for id, data in cam1_table.items():
    #     # n???u id ko trong ?????ng id v???a nh???n ???????c
    #     if id not in ls_id_received:
    #         # t??ng s??? frame t??? l???n nh??n th???y g???n nh???t
    #         data['num_frame_from_last_time_see'] += 1
    #         data['num_frame_from_last_time_update'] += 1

    #         # n???u s??? frame t??? l???n nh??n th???y g???n nh???t l???n h??n ng?????ng cho ph??p => x??a ?????i t?????ng n??y
    #         if data['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:

    #             # get doi tuong tuong ung trong bang cam 2
    #             if id in cam1_to_cam2_final:
    #                 correspond_cam2_id = cam1_to_cam2_final[id]
    #                 correspond_cam2_object = cam1_table[correspond_cam2_id]

    #                 # n???u s??? frame t??? l???n nh??n th???y g???n nh???t ??? cam 2 c??ng l???n h??n ng?????ng cho ph??p => x??a ?????i t?????ng n??y
    #                 if correspond_cam2_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
    #                     come_out_time = max(data['last_time_see'], correspond_cam2_object['last_time_see'])
    #                     come_in_time = min(data['first_time_see'], correspond_cam2_object['first_time_see'])
    #                     stay_time = come_out_time - come_in_time
    #                     global_ls_stay_time.append(stay_time)

    #                     ls_delete.append(id)

    #             else:  # n???u ko c?? ?????i t?????ng t????ng ???ng trong b???ng cam 2
    #                 come_out_time = data['last_time_see']
    #                 come_in_time = data['first_time_see']
    #                 stay_time = come_out_time - come_in_time
    #                 global_ls_stay_time.append(stay_time)

    #                 ls_delete.append(id)


    # # x??a nh???ng ?????i t?????ng ???? x??c ?????nh ra kh???i b???ng
    # for id in ls_delete:
    #     # xoa doi tuong tuong ung o bang cam 2
    #     try:
    #         del cam2_table[cam1_to_cam2_final[id]]
    #     except:
    #         pass
        
    #     # xoa doi tuong o bang cam 1
    #     del cam1_table[id]

    #     # xoa doi tuong o bang cam 1 to cam 2 counter
    #     try:
    #         ls_keys = list(cam1_to_cam2_counter.keys())
    #         for key in ls_keys:
    #             if key[0] == id:
    #                 del cam1_to_cam2_counter[key]
    #     except:
    #         pass
        
    #     # xoa doi tuong o bang cam 1 to cam 2 final
    #     try:
    #         del cam1_to_cam2_final[id]
    #     except:
    #         pass


    # print()
    # print('------- update performed on cam 1 -------')
    # print()
    # print('CAM 1 TABLE:')
    # show_table(cam1_table)
    # print()
    # print('CAM 2 TABLE:')
    # show_table(cam2_table)
    # print()
    # print('cam1_to_cam2_counter: ', cam1_to_cam2_counter)
    # print('cam1_to_cam2_final: ', cam1_to_cam2_final)
    # print('global ls stay time: ', global_ls_stay_time)
    # print()
    show_table()
    return {'message': 'data received from cam1'}




# endpoint to receive data from camera 2
@app.post('/cam2')
async def cam2(item_list: ItemList):
    # duyet qua toan bo item
    # neu khong co doi tuong nao trong khung hinhf => cam 2 gui len 1 list rong => se ko lam gi ca
    for item in item_list.__root__:
        if item.id == -1:
            continue
        # neu id gui len da co o trong bang
        if item.id in cam2_table.keys():
            myobject = cam2_table[item.id]
            if myobject['num_frame_from_last_time_update'] >= MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE and len(myobject['features']) < MAX_NUM_FEATURES:
                myobject['features'].append(item.feature)
                myobject['num_frame_from_last_time_update'] = -1

            myobject['last_time_see'] = item.time
            myobject['num_frame_from_last_time_see'] = 0
            myobject['num_frame_from_last_time_update'] += 1

        else:  # neu id gui len chua co o trong bang (la doi tuong moi)
            # them doi tuong vao bang
            cam2_table[item.id] = {
                'first_time_see': item.time,
                'features': [item.feature],
                'last_time_see': item.time,
                'num_frame_from_last_time_see': 0,
                'num_frame_from_last_time_update': 0
            }
            # print(f'ID {item.id} entered cam 2 at {time2datetime(item.time)}')

        # check feature hien tai cua item nay co giong voi feature nao trong bang cam 1 khong (c?? th??? c??c l???n kh??c ch??a gi???ng nh??ng l???n n??y l???i gi???ng)
        min_id = check_cosine_similarity(item.feature, cam1_table)
        if min_id != -1:  # n???u c??
            # n???u id n??y ch??a xu???t hi???n trong cam2_to_cam1_final (l???n ?????u ti??n c?? c??i gi???ng n?? ??? cam 1), th??m n?? v??o
            if item.id not in cam2_to_cam1_final:
                cam2_to_cam1_final[item.id] = min_id
                cam1_to_cam2_counter[(min_id, item.id)] = 1
                # print(f'ID {item.id} in cam 2 is now matched with ID {min_id} in cam 1')

            # n???u c???p (min_id, id) n??y ???? c?? t??? tr?????c ????, t??ng ????? c???ng c??? 2 c??i n??y l?? m???t l??n 1
            if (min_id, item.id) in cam1_to_cam2_counter:
                cam1_to_cam2_counter[(min_id, item.id)] += 1 # (cam1_id, cam2_id)
                # c???p nh???t id trong cam 1 gi???ng n?? nh???t
                if cam1_to_cam2_counter[(min_id, item.id)] > cam1_to_cam2_counter[(cam2_to_cam1_final[item.id], item.id)]:
                    cam2_to_cam1_final[item.id] = min_id
                    # print(f'ID {item.id} in cam 2 is now matched with ID {min_id} in cam 1')

            else: # n???u c???p (min_id, id) n??y ch??a c?? => th??m v??o 
                cam1_to_cam2_counter[(min_id, item.id)] = 1
        
        
    # # save cam2_table to json
    # with open('cam2_mbnv2_128x64_trt.json', 'w') as f:
    #     json.dump(cam2_table, f)

    # duyet qua tat ca nhung id khong xuat hien o lan nhan nay trong bang cam 2
    ls_id_received = [item.id for item in item_list.__root__]
    ls_delete = []
    for id, cam2_object in cam2_table.items():
        # n???u id ko trong ?????ng id v???a nh???n ???????c
        if id not in ls_id_received:
            # t??ng s??? frame t??? l???n nh??n th???y g???n nh???t
            cam2_object['num_frame_from_last_time_see'] += 1
            cam2_object['num_frame_from_last_time_update'] += 1

            # n???u s??? frame t??? l???n nh??n th???y g???n nh???t l???n h??n ng?????ng cho ph??p => x??a ?????i t?????ng n??y
            if cam2_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                # get doi tuong tuong ung trong bang cam 1
                if id in cam2_to_cam1_final:
                    correspond_cam1_id = cam2_to_cam1_final[id]
                    correspond_cam1_object = cam1_table[correspond_cam1_id]
                    # n???u s??? frame t??? l???n nh??n th???y g???n nh???t ??? cam 1 c??ng l???n h??n ng?????ng cho ph??p => x??a ?????i t?????ng n??y
                    # if correspond_cam1_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                    if True:
                        come_out_time = max(cam2_object['last_time_see'], correspond_cam1_object['last_time_see'])
                        come_in_time = min(cam2_object['first_time_see'], correspond_cam1_object['first_time_see'])
                        stay_time = come_out_time - come_in_time
                        global_ls_stay_time[id] = stay_time


                        ls_delete.append(id)

                else:  # n???u ko c?? ?????i t?????ng t????ng ???ng trong b???ng cam 1
                    come_out_time = cam2_object['last_time_see']
                    come_in_time = cam2_object['first_time_see']
                    stay_time = come_out_time - come_in_time
                    global_ls_stay_time[id] = stay_time

                    ls_delete.append(id)
                
                # print(global_ls_stay_time)
                # print(cam2_to_cam1_final)
    
    # xoa
    # for id in ls_delete:
    #     # xoa object tuong ung trong bang cam 1
    #     try:
    #         del cam1_table[cam2_to_cam1_final[id]]
    #     except:
    #         pass
        
    #     # xoa object trong cam 2
    #     del cam2_table[id]

    #     # xoa object tuong ung trong cam1_to_cam2_counter
    #     try:
    #         ls_keys = list(cam1_to_cam2_counter.keys())
    #         for key in ls_keys:
    #             if key[0] == cam2_to_cam1_final[id]:
    #                 del cam1_to_cam2_counter[key]
    #     except:
    #         pass

    #     try:
    #         del cam2_to_cam1_final[id]
    #     except:
    #         pass

    # print()
    # print('------- update performed on cam 2 -------')
    # print()
    # print('CAM 1 TABLE:')
    # show_table(cam1_table)
    # print()
    # print('CAM 2 TABLE:')
    # show_table(cam2_table)
    # print()
    # print('cam1_to_cam2_counter: ', cam1_to_cam2_counter)
    # print('cam2_to_cam1_final: ', cam2_to_cam1_final)
    # print('global ls stay time: ', global_ls_stay_time)
    # print()
    show_table()
    # print('cam2 to cam1 final: ', cam2_to_cam1_final)
    return {'message': 'data received from cam2'}


if __name__ == '__main__':
    uvicorn.run('server:app', host='localhost', port=8001, reload=True, log_level='info')
