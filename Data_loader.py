import copy
import math
import time
from cleverhans.future.torch.attacks.fast_gradient_method import fast_gradient_method
import matplotlib.pyplot as plt
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Test import ClassificationModel
from torch.autograd import Variable
from DataSet import ImageDataset

# FGSM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

total_dataset = ImageDataset(30)
# total_dataset = torch.load('total_dataset.pt')
test_loader = DataLoader(total_dataset, batch_size = 8, shuffle = True, drop_last = False)

pretrained_model = ClassificationModel()
pretrained_model.load_state_dict(torch.load('checkpoint.pth'))
pretrained_model.to(device)
epsilon = 0.1
criterion = torch.nn.CrossEntropyLoss()
accuracies = []
examples = []

# example
# data = next(iter(test_loader))
# img1, img2, label = data
# im = img2[0]
# plt.imshow(im.permute([1, 2, 0]))
# plt.show()
# for d in data:
#     try:
#         d.requires_grad = True
#     except:
#         pass
# img1, img2, label = data
# img1 = img1.to(device)
# img2 = img2.to(device)
# label = label.to(device)
# output = pretrained_model(img2)
# loss = criterion(output, label)
# pretrained_model.zero_grad()
# loss.backward()
# data_grad = img2.grad.data
# new_data = fgsm_attack(img2, epsilon, data_grad)
#
# nd = new_data[0].to('cpu')
# var_no_grad = nd.detach()
# plt.imshow(var_no_grad.permute([1, 2, 0]))
# plt.show()


def evalAdvAttack(fgsm_model=None, test_loader=None):
    print("Evaluating single model results on adv data")
    total = 0
    correct = 0
    fgsm_model.eval()
    for xs, ys in test_loader:
      if torch.cuda.is_available():
        xs, ys = xs.cuda(), ys.cuda()
      #pytorch fast gradient method
      xs = fast_gradient_method(fgsm_model, xs, eps=0.1, norm=np.inf, clip_min=0., clip_max=1.)
      # xs = fast_gradient_method(fgsm_model, xs, eps=0.1, norm=np.inf)
      xs, ys = Variable(xs), Variable(ys)
      preds1 = fgsm_model(xs)
      preds_np1 = preds1.cpu().detach().numpy()
      finalPred = np.argmax(preds_np1, axis=1)
      correct += (finalPred == ys.cpu().detach().numpy()).sum()
      total += test_loader.batch_size
    acc = float(correct) / total
    print('Adv accuracy: {:.3f}％'.format(acc * 100))

import numpy as np
data = next(iter(test_loader))
pretrained_model = ClassificationModel()
pretrained_model.load_state_dict(torch.load('checkpoint.pth'))
pretrained_model.to(device)
img1, img2, label = data
img1 = img1.cuda()
img2 = img2.cuda()
label = label.cuda()
xs = fast_gradient_method(pretrained_model, img2, eps=epsilon, norm=np.inf, clip_min=0., clip_max=1.)
print(xs.shape)

nd = xs[0].to('cpu')
var_no_grad = nd.detach()
plt.imshow(var_no_grad.permute([1, 2, 0]))
plt.show()


import numpy as np
a = np.ones(10)
b = np.ones(10)
print(a+ b)