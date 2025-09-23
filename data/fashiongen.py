

from torch.utils.data.dataset import Dataset
import h5py
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

# Full candidate Retrieval 

__all__ = ["FashionGen"]


class FashionGen(Dataset):
    def __init__(self, data_root, mode='train', image_size=224):
        assert mode in ['train', 'validation']

        self.data_root = data_root
        self.mode = mode

        # Load Data
        data_name = os.path.join(data_root, 'fashiongen_256_256_{}.h5'.format(mode))    # fashiongen_256_256_train.h5

        all_data = h5py.File(data_name, mode='r')
        self.images = all_data['input_image']
        self.text = all_data['input_description']

        # =========================== Tags ===========================
        self.category = all_data['input_subcategory']
        self.brand = all_data['input_brand']
        self.season = all_data['input_season']
        self.composition = all_data['input_composition']

        # =========================== Tags summary ===========================
        self.cat_id = {}
        for line in open(os.path.join(data_root, 'cat_id.txt')):
            k, v = line.strip().split(',')
            self.cat_id[k] = int(v)    

        self.brand_id = {}
        for line in open(os.path.join(data_root, 'brand_id.txt')):
            k, v = line.strip().split(',')
            self.brand_id[k] = int(v)        

        self.season_id = {}
        for line in open(os.path.join(data_root, 'season_id.txt')):
            k, v = line.strip().split(',')
            self.season_id[k] = int(v)

        self.comp_id = {}
        for line in open(os.path.join(data_root, 'comps_id.txt')):
            k, v = line.strip().split('&&')
            self.comp_id[k] = int(v)   
        

        # =========================== all product ===========================
        self.productIDs = all_data['input_productID']
        self.all_products = {}                      # {86605:[0,1,2,3], }       # only 60K products
        for idx in range(len(self.productIDs)):     # 260K
            p_id = self.productIDs[idx][0]          # int
            if p_id not in self.all_products:
                self.all_products[p_id] = [idx]
            else:
                self.all_products[p_id].append(idx)

        self.product_lists = list(self.all_products)    # save all keys
        print('Total number of Products {}'.format(len(self.product_lists)))

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
        product_id = self.product_lists[pidx]
        img_id_list = self.all_products[product_id]              # a list [0,1,2,3]

        # ======================= Image =======================
        if self.mode == 'validation':
            img_id = img_id_list[0]
        else:
            img_id = np.random.choice(img_id_list)               # random choose one image
        
        # Load image
        image = self.images[img_id]                   # RGB numpy array, (256, 256, 3), val in 0-255
        image = Image.fromarray(image)              # Pillow Image
        image = self.image_transform(image)              # (3, 224, 224) tensor normalized

        # ======================= Text =======================
        text_id = img_id_list[0]  # all images have the same ID
        text = self.text[text_id][0].decode('latin1')   # string

        # ======================= tags =======================
        category_str = self.category[text_id][0].decode('latin1')       # string
        brand_str = self.brand[text_id][0].decode('latin1')             # string
        season_str = self.season[text_id][0].decode('latin1')           # string
        comp_str = self.composition[text_id][0].decode('latin1')        # sting

        category_id = self.cat_id[category_str]     # int
        brand_id = self.brand_id[brand_str]         # int
        season_id = self.season_id[season_str]      # int
        comp_id = self.comp_id[comp_str]            # int

        # tags_str = {
        #     'category': category_str, 
        #     'brand': brand_str, 
        #     'season': season_str, 
        #     'composition': comp_str
        # }

        tags_id = {
            'category': category_id, 
            'brand': brand_id, 
            'season':season_id, 
            'composition': comp_id
        }
        text_raw = comp_str + '.' + category_str + '.' + brand_str + '.' + season_str + '.' + text
        return image, text, text_raw, tags_id


if __name__ == '__main__':

    import clip
    device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device)

    data = FashionGen(data_root='/Users/mahaoyu/data/multimodal/fashiongen/',
                          mode='validation')

    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    img, txt = batch
    print(img.shape)
    print(torch.min(img), torch.max(img))
    print(len(txt))





