'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics
import os
from tqdm import tqdm
import cv2

# if not os.path.exists('./output_img'):
#     os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

vis_path = '.tmp'+ '/vis/' + opt.dataset + '_' + opt.backbone + '_'+ opt.mode

if not os.path.exists(vis_path):
    os.makedirs(vis_path)


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if opt.dataset == 'cdd':
    opt.dataset_dir = '../Dataset/cdd_dataset/'
elif opt.dataset == 'levir':
	opt.dataset_dir = '../Dataset/levir_patch_dataset/'

test_loader = get_test_loaders(opt, batch_size=1)

# path = 'weights/snunet-32.pt'   # the path of the model
# model = torch.load(path)

model = torch.load(opt.path, map_location='cpu')

model.to(dev)


model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, fname in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)

        #cd_preds = cd_preds[-1] # BIT输出不是tuple
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        #file_path = './output_img/' + str(index_img).zfill(5)
        cv2.imwrite(vis_path +'/'+ fname[0], cd_preds)

        index_img += 1
