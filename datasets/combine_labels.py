from typing import Optional, List, Set, Union


class CombineLabels:
    """
    For a multi-class classification problem like action classification, we might be interested in
    combining related labels into a new "super category".

    CombineLabels stores data about groups of labels to combine, and ones to remove.
    Its attribute `action_to_idx` maps individual actions to the appropriate super category index.
    """
    def __init__(self, category_groups: List[Union[Set, str]], to_remove: Optional[Set] = None):
        self.category_groups = category_groups
        if to_remove is None:
            self.to_remove = set()
        else:
            self.to_remove = to_remove

        # Store mapping from the individual action names (initial categories) to the index of
        # the resultant super category.
        self.action_to_idx = {}
        # list of final string category names. When categories are combined into super category,
        # the names of the individual categories are joined.
        # E.g., "run" and "walk" combine to "run + walk."
        self.final_category_names = []
        current_idx = 0
        for s in self.category_groups:
            if isinstance(s, set):
                for action in s:
                    self._validate_not_to_remove_action(action)
                    self.action_to_idx[action] = current_idx
                self.final_category_names.append(" + ".join(sorted(s)))
            elif isinstance(s, str):
                self._validate_not_to_remove_action(s)
                self.action_to_idx[s] = current_idx
                self.final_category_names.append(s)
            else:
                raise TypeError(f"{s} is of unsupported type {type(s)}. Must be string or set.")
            current_idx += 1

    def _validate_not_to_remove_action(self, action):
        if action in self.to_remove:
            raise ValueError(
                f"{action} in set to be removed {self.to_remove}; can't also be in label group."
            )

    def validate_against(self, master_verbs):
        # Ensure that all the sets are disjoint, and that they exhaustively cover
        # everything in master_verbs
        union = set()
        total_len = 0
        for s in self.category_groups:
            if isinstance(s, str):
                s = {s}
            total_len += len(s)
            union |= s
        total_len += len(self.to_remove)
        union |= self.to_remove
        if len(union) != total_len:
            raise ValueError("Overlapping sets!")
        set_master_verbs = set(master_verbs)
        if union != set_master_verbs:
            raise ValueError(f"{sorted(union)} != {sorted(set_master_verbs)}")

    def get_final_idx(self, verb):
        return self.action_to_idx.get(verb)
