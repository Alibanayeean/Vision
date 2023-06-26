import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class_to_id = {
    "bird": 0,
    "cat": 1,
    "cow": 2,
    "dog": 3,
    "elephant": 4,
    "giraffe": 5,
    "horse": 6,
    "zebra": 7,
}

id_to_class = {class_to_id[class_name]: class_name for class_name in class_to_id.keys()}

class ImageDataset(Dataset):
    def __init__(self, num_samples = None):
        path = './Data/Animals/train/'
        self.size = 224
        self.images = []
        self.lables = []

        trans_to_tensor = transforms.ToTensor()
        self.augmentation_train = transforms.Compose([
                transforms.RandomResizedCrop(self.size, (0.6, 1)),
                transforms.RandomHorizontalFlip()
        ])
        self.augmentation_test = transforms.Compose([
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
        ])

        trans = transforms.Resize(self.size)

        for class_name in class_to_id.keys():
            for idx, I in tqdm(enumerate(os.listdir(path + class_name))):
                img = Image.open(path + class_name + '/' + I)
                tensor_img = trans_to_tensor(img)
                if len(tensor_img) == 1:
                    tensor_img = torch.cat([tensor_img, tensor_img, tensor_img])
                tensor_img = trans(tensor_img)
                self.images.append(tensor_img)

                self.lables.append(class_to_id[class_name])


                if num_samples != None and idx >= num_samples - 1:
                    break
        self.lables = torch.tensor(self.lables, dtype = torch.long)

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, index):
        tensor_img_train = self.augmentation_train(self.images[index])
        tensor_img_test = self.augmentation_test(self.images[index])
        return tensor_img_train, tensor_img_test, self.lables[index]

if __name__ == '__main__':
    total_dataset = ImageDataset(10)
    tr = total_dataset.__getitem__(52)
    plt.imshow(tr[0].permute([1, 2, 0]))
    plt.show()
    plt.imshow(tr[0].permute([1, 2, 0]))
    plt.show()
    plt.imshow(tr[0].permute([1, 2, 0]))
    plt.show()
    plt.imshow(tr[0].permute([1, 2, 0]))
    plt.show()
    # load = False
    # if load:
    #     total_dataset = torch.load('total_dataset.pt')
    # else:
    #     # total_dataset = ImageDataset()
    #     torch.save(total_dataset, 'total_dataset.pt')
    #     print('finish')
    #     exit()
    #
    #
    #
