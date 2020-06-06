import json

import numpy as np
from .combine_labels import CombineLabels


class VCOCO:
    def __init__(self, path_to_vcoco_anns: str, combine_labels: CombineLabels = None):
        with open(path_to_vcoco_anns, "r") as f:
            self.vcoco_anns = json.load(f)
        self.ann_labels = None
        self.img_to_ann_ids = None
        self.ann_to_img_id = None
        self.ann_ids = None
        self.img_ids = None
        self.combine_labels = combine_labels
        self._parse_vcoco_anns()

    def _parse_vcoco_anns(self):
        all_actions = [el["action_name"] for el in self.vcoco_anns]  # master list of all actions
        if self.combine_labels is None:
            self.combine_labels = CombineLabels(all_actions)
        self.combine_labels.validate_against(all_actions)

        for idx, el in enumerate(self.vcoco_anns):
            action = el["action_name"]
            d = {key: np.asarray(val) for key, val in el.items() if key != "action_name"}
            if idx == 0:
                self.img_to_ann_ids = {
                    img_id: set(d["ann_id"][d["image_id"] == img_id]) for img_id in set(d["image_id"])
                }
                self.ann_to_img_id = dict(zip(d["ann_id"], d["image_id"]))
                self.ann_ids = d["ann_id"]
                self.img_ids = d["image_id"]
                self.ann_labels = np.zeros((self.n_classes, self.n_annotations))
            combine_action_idx = self.combine_labels.get_final_idx(action)  # "super category" id
            if combine_action_idx is not None:
                # Set label to 1 if the annotation depicts any action in a particular category
                self.ann_labels[combine_action_idx, :] = np.maximum(
                    self.ann_labels[combine_action_idx, :], d["label"]
                )

    @property
    def n_classes(self):
        return len(self.category_names)

    @property
    def n_annotations(self):
        return len(self.ann_ids)

    @property
    def n_images(self):
        return len(set(self.img_ids))

    @property
    def category_names(self):
        return self.combine_labels.final_category_names
