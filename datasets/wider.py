import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import random

from .augmentation import preprocess


class WIDERDetection(data.Dataset):
    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, targets, h, w = self.pull_item(index)
        target = {}
        bbox =[]
        label = []
        for i in range(targets.shape[0]):
            bbox.append(targets[0][:4])
            label.append(targets[0][-1])
        target['boxes'] = torch.as_tensor(bbox, dtype=torch.float32)
        target['labels'] = torch.as_tensor(label, dtype=torch.int64)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        return torch.from_numpy(img), target, im_height, im_width
        
    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def build(image_set, args):
    if image_set == 'train':
        dataset = WIDERDetection('./datasets/train.txt', mode='train')
        return dataset
    elif image_set == 'val':
        dataset = WIDERDetection('./datasets/val.txt', mode='val')
        return dataset
