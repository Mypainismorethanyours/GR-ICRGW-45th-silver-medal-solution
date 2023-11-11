import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class NpyDataset(CustomDataset):
    def load_annotations(self):
        # 加载图像和标签的文件路径列表
        image_paths = self.img_prefix / self.img_infos[self.split]['img_dir']
        label_paths = self.img_prefix / self.img_infos[self.split]['ann_dir']
        data_infos = []
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            data_infos.append(dict(img_path=str(img_path), lbl_path=str(lbl_path), img_shape=self.img_shapes[i]))
        return data_infos

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        img_path = img_info['img_path']
        lbl_path = img_info['lbl_path']

        # 从npy文件中加载图像数据
        img = np.load(img_path)

        # 加载标签数据，如果需要的话
        lbl = self.load_mask(lbl_path)

        results = dict(img=img, lbl=lbl)

        return self.pipeline(results)