import torch
from bresenham import bresenham
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def collate_self_train(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'positive_img': [], 'positive_boxes': [],
                 'negative_img': [], 'negative_boxes': [],
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())
        batch_mod['negative_boxes'].append(torch.tensor(i_batch['negative_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)

    return batch_mod


def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                 'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod


def mydrawPNG(vector_image , data_points):
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    final_list = []
    for i in range( 0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        final_list.extend([list(j) for j in cordList])
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])
    random.shuffle(final_list)
    final_list = final_list[:data_points]
    return final_list
