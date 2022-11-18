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

# define global variables
cam1_table, cam2_table = {}, {}
MAX_NUM_FRAME_FROM_LAST_TIME_UPDATE = -1
MAX_NUM_FEATURES = 50
MAX_NUM_FRAME_FROM_LAST_TIME_SEE = 40
L2_SIMILARITY_THRESHOLD = 1e6
COSINE_SIMILARITY_THRESHOLD = 1e6
MAX_CHECK_OBSOLETE_INTERVAL = MAX_NUM_FRAME_FROM_LAST_TIME_SEE * 2
cam1_to_cam2_counter = {}
cam1_to_cam2_final, cam2_to_cam1_final = {}, {}
global_ls_stay_time = []
check_obsolete_counter_1, check_obsolete_counter_2 = 0, 0

# util functions
def show_table(table: dict):
    for id, data in table.items():
        ls_len_features = [len(el) for el in data['features']]
        print(f"ID: {id}, Data: {ls_len_features}")

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


def check_feature_similarity(input_feature: list, table: dict, metric='l2', threshold=L2_SIMILARITY_THRESHOLD) -> int:
    # tim doi tuong giong voi doi tuong nay nhat ở bảng table
    min_distance = 1e9
    min_id = -1
    similarity_func = l2_similarity if metric == 'l2' else cosine_similarity
    for id, data in table.items():
        if id == -1:
            continue
        for feature in data['features']:
            distance = similarity_func(input_feature, feature)
            if distance < min_distance:
                min_distance =  distance
                min_id = id
    
    # nếu khoảng cách gần nhất nhỏ hơn ngưỡng cho phép => 2 đối tượng này là 1
    if min_distance < threshold:
        return min_id
    else:
        return -1



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
            print(f'ID {item.id} entered cam 1 at {item.time}')
    # save cam1_table to json
    # with open('cam1.json', 'w') as f:
    #     json.dump(cam1_table, f)

    #     # check feature hien tai cua item nay co giong voi feature nao trong bang cam 2 khong (có thể các lần khác chưa giống nhưng lần này lại giống)
    #     min_id = check_feature_similarity(item.feature, cam2_table, metric='l2', threshold=L2_SIMILARITY_THRESHOLD)
    #     if min_id != -1:  # nếu có
    #         # nếu id này chưa xuất hiện trong cam1_to_cam2_final (lần đầu tiên có cái giống nó ở cam 2), thêm nó vào
    #         if item.id not in cam1_to_cam2_final:
    #             cam1_to_cam2_final[item.id] = min_id

    #         # nếu cặp (min_id, id) này đã có từ trước đó, tăng độ củng cố 2 cái này là một lên 1
    #         if (item.id, min_id) in cam1_to_cam2_counter:
    #             cam1_to_cam2_counter[(item.id, min_id)] += 1 # (cam1_id, cam2_id)
    #             # cập nhật id trong cam 2 giống nó nhất
    #             if cam1_to_cam2_counter[(item.id, min_id)] > cam1_to_cam2_counter[(item.id, cam1_to_cam2_final[item.id])]:
    #                 cam1_to_cam2_final[item.id] = min_id
    #         else: # nếu cặp (min_id, id) này chưa có => thêm vào 
    #             cam1_to_cam2_counter[(item.id, min_id)] = 1

    # # duyet qua tat ca nhung id khong xuat hien o lan nhan nay trong bang cam 1
    # ls_id_received = [item.id for item in item_list.__root__]
    # ls_delete = []
    # for id, data in cam1_table.items():
    #     # nếu id ko trong đống id vừa nhận được
    #     if id not in ls_id_received:
    #         # tăng số frame từ lần nhìn thấy gần nhất
    #         data['num_frame_from_last_time_see'] += 1
    #         data['num_frame_from_last_time_update'] += 1

    #         # nếu số frame từ lần nhìn thấy gần nhất lớn hơn ngưỡng cho phép => xóa đối tượng này
    #         if data['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:

    #             # get doi tuong tuong ung trong bang cam 2
    #             if id in cam1_to_cam2_final:
    #                 correspond_cam2_id = cam1_to_cam2_final[id]
    #                 correspond_cam2_object = cam1_table[correspond_cam2_id]

    #                 # nếu số frame từ lần nhìn thấy gần nhất ở cam 2 cũng lớn hơn ngưỡng cho phép => xóa đối tượng này
    #                 if correspond_cam2_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
    #                     come_out_time = max(data['last_time_see'], correspond_cam2_object['last_time_see'])
    #                     come_in_time = min(data['first_time_see'], correspond_cam2_object['first_time_see'])
    #                     stay_time = come_out_time - come_in_time
    #                     global_ls_stay_time.append(stay_time)

    #                     ls_delete.append(id)

    #             else:  # nếu ko có đối tượng tương ứng trong bảng cam 2
    #                 come_out_time = data['last_time_see']
    #                 come_in_time = data['first_time_see']
    #                 stay_time = come_out_time - come_in_time
    #                 global_ls_stay_time.append(stay_time)

    #                 ls_delete.append(id)


    # # xóa những đối tượng đã xác định ra khỏi bảng
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
    return {'message': 'data received from cam1'}




# endpoint to receive data from camera 2
@app.post('/cam2')
async def cam2(item_list: ItemList):
    # duyet qua toan bo item
    # neu khong co doi tuong nao trong khung hinhf => cam 2 gui len 1 list rong => se ko lam gi ca
    for item in item_list.__root__:
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
            print(f'ID {item.id} entered cam 2 at {item.time}')

        # check feature hien tai cua item nay co giong voi feature nao trong bang cam 1 khong (có thể các lần khác chưa giống nhưng lần này lại giống)
        min_id = check_feature_similarity(item.feature, cam1_table, metric='l2', threshold=L2_SIMILARITY_THRESHOLD)
        if min_id != -1:  # nếu có
            # nếu id này chưa xuất hiện trong cam2_to_cam1_final (lần đầu tiên có cái giống nó ở cam 1), thêm nó vào
            if item.id not in cam2_to_cam1_final:
                cam2_to_cam1_final[item.id] = min_id
                cam1_to_cam2_counter[(min_id, item.id)] = 1
                print(f'ID {item.id} in cam 2 is now matched with ID {min_id} in cam 1')

            # nếu cặp (min_id, id) này đã có từ trước đó, tăng độ củng cố 2 cái này là một lên 1
            if (min_id, item.id) in cam1_to_cam2_counter:
                cam1_to_cam2_counter[(min_id, item.id)] += 1 # (cam1_id, cam2_id)
                # cập nhật id trong cam 1 giống nó nhất
                if cam1_to_cam2_counter[(min_id, item.id)] > cam1_to_cam2_counter[(cam2_to_cam1_final[item.id], item.id)]:
                    cam2_to_cam1_final[item.id] = min_id
                    print(f'ID {item.id} in cam 2 is now matched with ID {min_id} in cam 1')

            else: # nếu cặp (min_id, id) này chưa có => thêm vào 
                cam1_to_cam2_counter[(min_id, item.id)] = 1
        
        
    # # save cam2_table to json
    # with open('cam2.json', 'w') as f:
    #     json.dump(cam2_table, f)

    # duyet qua tat ca nhung id khong xuat hien o lan nhan nay trong bang cam 2
    ls_id_received = [item.id for item in item_list.__root__]
    ls_delete = []
    for id, cam2_object in cam2_table.items():
        # nếu id ko trong đống id vừa nhận được
        if id not in ls_id_received:
            # tăng số frame từ lần nhìn thấy gần nhất
            cam2_object['num_frame_from_last_time_see'] += 1
            cam2_object['num_frame_from_last_time_update'] += 1

            # nếu số frame từ lần nhìn thấy gần nhất lớn hơn ngưỡng cho phép => xóa đối tượng này
            if cam2_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                # get doi tuong tuong ung trong bang cam 1
                if id in cam2_to_cam1_final:
                    correspond_cam1_id = cam2_to_cam1_final[id]
                    correspond_cam1_object = cam1_table[correspond_cam1_id]
                    # nếu số frame từ lần nhìn thấy gần nhất ở cam 1 cũng lớn hơn ngưỡng cho phép => xóa đối tượng này
                    # if correspond_cam1_object['num_frame_from_last_time_see'] >= MAX_NUM_FRAME_FROM_LAST_TIME_SEE:
                    if True:
                        come_out_time = max(cam2_object['last_time_see'], correspond_cam1_object['last_time_see'])
                        come_in_time = min(cam2_object['first_time_see'], correspond_cam1_object['first_time_see'])
                        stay_time = come_out_time - come_in_time
                        global_ls_stay_time.append(stay_time)

                        ls_delete.append(id)

                else:  # nếu ko có đối tượng tương ứng trong bảng cam 1
                    come_out_time = cam2_object['last_time_see']
                    come_in_time = cam2_object['first_time_see']
                    stay_time = come_out_time - come_in_time
                    global_ls_stay_time.append(stay_time)

                    ls_delete.append(id)
    
    # xoa
    for id in ls_delete:
        # xoa object tuong ung trong bang cam 1
        try:
            del cam1_table[cam2_to_cam1_final[id]]
        except:
            pass
        
        # xoa object trong cam 2
        del cam2_table[id]

        # xoa object tuong ung trong cam1_to_cam2_counter
        try:
            ls_keys = list(cam1_to_cam2_counter.keys())
            for key in ls_keys:
                if key[0] == cam2_to_cam1_final[id]:
                    del cam1_to_cam2_counter[key]
        except:
            pass

        try:
            del cam2_to_cam1_final[id]
        except:
            pass

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
    return {'message': 'data received from cam2'}


if __name__ == '__main__':
    uvicorn.run('server:app', host='localhost', port=8001, reload=True, log_level='critical')
