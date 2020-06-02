import json

import numpy as np


class VCOCO:
    def __init__(self, path_to_vcoco_anns: str):
        with open(path_to_vcoco_anns, "r") as f:
            self.vcoco_anns = json.load(f)
        self.category_names = None
        self.ann_labels = None
        self.img_to_ann_ids = None
        self.ann_to_img_id = None
        self.ann_ids = None
        self.img_ids = None
        self.parse_vcoco_anns()

    def parse_vcoco_anns(self):
        self.category_names = np.asarray([])
        for idx, el in enumerate(self.vcoco_anns):
            d = {key: np.asarray(val) for key, val in el.items() if key != "action_name"}
            if idx == 0:
                self.img_to_ann_ids = {
                    img_id: set(d["ann_id"][d["image_id"] == img_id]) for img_id in set(d["image_id"])
                }
                self.ann_to_img_id = dict(zip(d["ann_id"], d["image_id"]))
                self.ann_ids = d["ann_id"]
                self.img_ids = d["image_id"]
            self.category_names = np.append(self.category_names, el["action_name"])
            if self.ann_labels is None:
                self.ann_labels = d["label"].reshape(1, -1)
            else:
                self.ann_labels = np.append(self.ann_labels, d["label"].reshape(1, -1), axis=0)

    @property
    def n_classes(self):
        return len(self.vcoco_anns)

    @property
    def n_annotations(self):
        return len(self.ann_ids)

    @property
    def n_images(self):
        return len(set(self.img_ids))
