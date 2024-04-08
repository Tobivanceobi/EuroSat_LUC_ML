import os
import time

import numpy as np
import rasterio
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from src.colors import bcolors
from src.feature_extraction import extract_hog_features, extract_color_hist

c = bcolors()


class EuroSatMS(Dataset):
    def __init__(self,
                 dataframe,
                 root_dir,
                 encoder,
                 feature_extractor,
                 select_chan=None,
                 n_jobs=-4):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.enc = encoder

        # If select_chan is None, use the RGB channels
        if select_chan is None:
            select_chan = [3, 2, 1]

        self.select_chan = select_chan

        print(f"\n{c.OKGREEN}Preloading images...{c.ENDC}")
        print(f"\n{c.OKCYAN}Images:         {len(dataframe)}{c.ENDC}")
        print(f"{c.OKCYAN}Jobs:           {n_jobs} {c.ENDC}\n")

        start_time = time.time()
        result = Parallel(n_jobs=n_jobs)(
            delayed(self.process_image)(idx) for idx in range(len(dataframe))
        )

        self.samples = []
        self.targets = []
        self.groups = []

        for x, y, idx in result:
            self.samples.append(x)
            self.targets.append(y)
            self.groups.append(idx)

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)
        end_time = time.time()
        t = end_time - start_time
        print(f"\n{c.OKBLUE}Time taken:      {int((t - (t % 60)) / 60)} min {t % 60} sec {c.ENDC}")

    def process_image(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])

        if ".npy" in img_path:
            image = np.load(img_path).transpose(2, 0, 1)
        else:
            with rasterio.open(img_path) as src:
                image = np.array(src.read())

        image = image[self.select_chan].astype(np.float32)

        # rgb_min, rgb_max = image.min(), image.max()
        # image = (image - rgb_min) / (rgb_max - rgb_min)
        # image = image.clip(0, 1)

        # minmax scale image channel wise
        for i in range(image.shape[0]):
            min_val, max_val = image[i].min(), image[i].max()
            image[i] = (image[i] - min_val) / (max_val - min_val)
            image[i] = image[i].clip(0, 1)

        image = (image * 255).astype(np.uint8)

        features = []
        for method in self.feature_extractor:
            if method == "hog":
                feats = extract_hog_features(image)
                feats = feats.flatten()
                features.extend(feats)
            elif method == "color_hist":
                feats = extract_color_hist(image)
                feats = feats.flatten()
                features.extend(feats)
            else:
                raise ValueError(f"Unknown feature extraction method: {method}")

        target = self.dataframe.iloc[idx, 1]
        target = self.enc.transform([target])[0]

        return features, target, idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self.samples[idx]
        image = image.astype(np.float32)

        target = self.targets[idx]

        return image, target
