

from torch.utils.data.dataset import Dataset
import json 
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

__all__ = ["AmazonFashion"]


class AmazonFashion(Dataset):
    def __init__(self, data_root, mode='train', image_size=224):
        assert mode in ['train', 'test']

        self.data_root = data_root
        self.mode = mode

        # =========================== Load labels ===========================
        data_name = os.path.join(data_root, 'meta-fashion.json')
        self.all_data = json.load(open(data_name))

        # =========================== all product ===========================
        self.all_products = json.load( open(os.path.join(data_root, 'split.json')))[mode]      # asins list
        print('Total number of Products {}'.format(len(self.all_products)))

        # =========================== Tags summary ===========================
        self.brand_id = {}
        for num, line in enumerate(open(os.path.join(data_root, 'brand_id.txt'))):
            try:
                k, v = line.strip().split(',')
                
            except:
                pass
#                 print('line', num, line.strip().split(','))
                items = line.strip().split(',')
                v = items[-1]
                k = ','.join(items[0:-1])
#                 print('key', num, k)
            
            try:
                self.brand_id[k] = int(v)

            except:
                pass
                

        # =========================== Transforms ===========================
        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=Image.BICUBIC),
            T.CenterCrop(image_size),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.all_products)

    def __getitem__(self, pidx):
        product_asin = self.all_products[pidx]            # asin: B00G5IMJWS
         
        meta_info = self.all_data[product_asin]         # dict, {'}

        # ======================= Image =======================
        img_names = meta_info['images']
        if self.mode == 'test':
            img = img_names[0]                          # [ "B00018I6Z2_0.png",  "B00018I6Z2_1.png"]
        else:
            img = np.random.choice(img_names)               # random choose one image

        # Load image
        img_path = os.path.join(self.data_root, 'Fashion', product_asin, img)
        image = Image.open(img_path)                    # Pillow Image        
        image = self.image_transform(image)              # (3, 224, 224) tensor normalized

        # ======================= Text =======================
        text = meta_info['title']

        # ======================= tags =======================
        brand_str = meta_info['brand']      # string
        try:
            brand_id = self.brand_id[brand_str]         # int
        except:
            brand_id = 99999999 # for UNKNOWN

        # original
#         tags_str = {
#             'brand': brand_str, 
#         }
        
        # modified version
        tags_str = brand_str + '. ' + text

        tags_id = {
            'brand': brand_id, 
        }
        
        # Return
        RACER = True
        if not RACER:
            return image, text, tags_str, tags_id
        else:
            #brand_str = 'a photo of an item made by B{}'.format(tags_id['brand']) # with prompt
            brand_str = brand_str
            return image, text, tags_str, tags_id, brand_str




