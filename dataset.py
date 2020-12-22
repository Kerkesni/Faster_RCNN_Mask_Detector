from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
import xml.etree.ElementTree as ET
import random
import torch

classes_index = {'with_mask':1, 'without_mask':2, 'mask_weared_incorrect':3} # 0 is for background!!

# Xml Labels Parser
def get_objects(xml_file):
  annotation = ET.parse(xml_file)
  root = annotation.getroot()
  objects = []
  for obj in root.findall('object'):
    new_object = {'name': obj.find('name').text}
    bbox_tree = obj.find('bndbox')
    new_object['bbox'] = [int(bbox_tree.find('xmin').text), int(bbox_tree.find('ymin').text), int(bbox_tree.find('xmax').text), int(bbox_tree.find('ymax').text)]
    objects.append(new_object)
  return objects

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #TODO: Add Data Augmentatio
    return T.Compose(transforms)

# Setting Up Dataset Class
class FaceMaskDataset(Dataset):
  def __init__(self, img_folder, annotation_folder, indexes, conversion=T.ToTensor()):
    self.conversion = conversion
    
    self.dataset = []
    for index in indexes:
      sample = {}
      sample['image'] = Image.open(os.path.join(img_folder, 'maksssksksss' + str(index) + '.png')).convert('RGB')
      sample['objects'] = get_objects(os.path.join(annotation_folder, 'maksssksksss' + str(index) + '.xml'))
      sample['id'] = index
      self.dataset.append(sample)
    
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    target = {'boxes': [], 'labels': []}
    for obj in self.dataset[idx]['objects']:
      target["boxes"].append(obj['bbox'])
      target['labels'].append(classes_index[obj['name']])
      
    target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
    target["labels"] = torch.as_tensor(target['labels'], dtype=torch.int64)
    target["image_id"] = torch.tensor([self.dataset[idx]['id']])

    img = self.dataset[idx]['image']
    if self.conversion is not None:
      img = self.conversion(img)
    return img, target

# Returns the training, validation and test datasets
def getDatasets(data_folder, train_split_percentage=0.5, val_split_percentage=0.3, test_split_percentage=0.2, size_of_the_dataset=853):

    indexes = list(range(size_of_the_dataset))
    random.shuffle(indexes)

    train_indexes = indexes[:int(train_split_percentage*len(indexes))]
    val_indexes = indexes[int(train_split_percentage*len(indexes)):int((train_split_percentage + val_split_percentage)*len(indexes))]
    test_indexes = indexes[int((train_split_percentage + val_split_percentage)*len(indexes)):]

    print(f"Effective train split = {len(train_indexes)/len(indexes)*100}%")
    print(f"Effective val split = {len(val_indexes)/len(indexes)*100}%")
    print(f"Effective test split = {len(test_indexes)/len(indexes)*100}%", flush=True)

    print("Loading training set")
    train_dataset = FaceMaskDataset(os.path.join(data_folder, 'images'), os.path.join(data_folder, 'annotations'), train_indexes, conversion=get_transform(True))
    print("Loading validation set")
    val_dataset = FaceMaskDataset(os.path.join(data_folder, 'images'), os.path.join(data_folder, 'annotations'), val_indexes, conversion=get_transform(False))
    print("Loading test set")
    test_dataset = FaceMaskDataset(os.path.join(data_folder, 'images'), os.path.join(data_folder, 'annotations'), test_indexes, conversion=get_transform(False))
    return train_dataset, val_dataset, test_dataset

# Returns the training, validation and test loaders
def getLoaders(train_dataset, val_dataset, batch_size=2):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader, test_dataset