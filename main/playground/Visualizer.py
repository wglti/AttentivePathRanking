import os
import numpy as np
import shutil

class Visualizer:

    def __init__(self, idx2entity, idx2entity_type, idx2relation, save_dir):
        self.idx2entity = idx2entity
        self.idx2entity_type = idx2entity_type
        self.idx2relation = idx2relation

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def visualize_paths(self, inputs, labels, type_weights, path_weights, rel, split, epoch):
        num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
        highest_weighted_type_indices = np.argmax(type_weights, axis=3)

        rel_dir = os.path.join(self.save_dir, rel)
        if not os.path.exists(rel_dir):
            os.mkdir(rel_dir)
        rel_split_dir = os.path.join(rel_dir, split)
        if not os.path.exists(rel_split_dir):
            os.mkdir(rel_split_dir)
        file_name = os.path.join(rel_split_dir, str(epoch) + ".detailed.tsv")

        with open(file_name, "a") as fh:
            for ent_pairs_idx in range(num_ent_pairs):
                paths = []
                subj = None
                obj = None
                label = labels[ent_pairs_idx]
                for path_idx in range(num_paths):
                    # Each path string should be: ent1[type1:weight1,...,typeC:weightC] - rel1 - ent2[type1:weight1,...,typeC:weightC]

                    # processing a path
                    path = []
                    start = False
                    for stp in range(num_steps):
                        feats = inputs[ent_pairs_idx, path_idx, stp]
                        entity = feats[-2]
                        entity_name = self.idx2entity[entity]
                        # ignore pre-paddings
                        if not start:
                            if entity_name != "#PAD_TOKEN":
                                start = True
                                if subj is None:
                                    subj = entity_name
                                else:
                                    assert subj == entity_name
                        if start:
                            rel = feats[-1]
                            types = feats[0:-2]
                            weights = type_weights[ent_pairs_idx, path_idx, stp]
                            types_str = []
                            for i in range(len(types)):
                                type_name = self.idx2entity_type[types[i]]
                                weight = weights[i]
                                type_str = type_name + ":" + "%.3f" % weight
                                types_str.append(type_str)
                            types_str = "[" + ",".join(types_str) + "]"
                            rel_name = self.idx2relation[rel]
                            path += [entity_name + types_str]
                            if rel_name != "#END_RELATION":
                                path += [rel_name]
                            if stp == num_steps - 1:
                                if obj is None:
                                    obj = entity_name
                                else:
                                    assert obj == entity_name
                    path_str = "-".join(path)
                    paths.append((path_str, path_weights[ent_pairs_idx, path_idx]))
                paths = sorted(paths, key=lambda x: x[1], reverse=True)
                weighted_paths = [p[0] + "," + str(p[1]) for p in paths]
                paths_str = " -#- ".join(weighted_paths)
                fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_str + "\n")

    def visualize_paths_with_relation_and_type(self, inputs, labels, type_weights, path_weights, rel, split, epoch):
        num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
        highest_weighted_type_indices = np.argmax(type_weights, axis=3)

        rel_dir = os.path.join(self.save_dir, rel)
        if not os.path.exists(rel_dir):
            os.mkdir(rel_dir)
        rel_split_dir = os.path.join(rel_dir, split)
        if not os.path.exists(rel_split_dir):
            os.mkdir(rel_split_dir)
        file_name = os.path.join(rel_split_dir, str(epoch) + ".tsv")

        with open(file_name, "a") as fh:
            for ent_pairs_idx in range(num_ent_pairs):
                paths = []
                subj = None
                obj = None
                label = labels[ent_pairs_idx]
                for path_idx in range(num_paths):
                    # Each path string should be: ent1[type1:weight1,...,typeC:weightC] - rel1 - ent2[type1:weight1,...,typeC:weightC]

                    # processing a path
                    path = []
                    start = False
                    for stp in range(num_steps):
                        feats = inputs[ent_pairs_idx, path_idx, stp]
                        entity = feats[-2]
                        entity_name = self.idx2entity[entity]
                        # ignore pre-paddings
                        if not start:
                            if entity_name != "#PAD_TOKEN":
                                start = True
                                if subj is None:
                                    subj = entity_name
                                else:
                                    assert subj == entity_name

                        if start:
                            rel = feats[-1]
                            types = feats[0:-2]
                            rel_name = self.idx2relation[rel]
                            highest_weighted_type = types[highest_weighted_type_indices[ent_pairs_idx, path_idx, stp]]
                            type_name = self.idx2entity_type[highest_weighted_type]
                            path += [type_name]
                            if rel_name != "#END_RELATION":
                                path += [rel_name]
                            if stp == num_steps - 1:
                                if obj is None:
                                    obj = entity_name
                                else:
                                    assert obj == entity_name
                    path_str = "-".join(path)
                    paths.append((path_str, path_weights[ent_pairs_idx, path_idx]))
                paths = sorted(paths, key=lambda x: x[1], reverse=True)
                weighted_paths = [p[0] + "," + str(p[1]) for p in paths]
                paths_str = " -#- ".join(weighted_paths)
                fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_str + "\n")

    def save_space(self, rel, best_epoch):
        rel_dir = os.path.join(self.save_dir, rel)
        for split in os.listdir(rel_dir):
            rel_split_dir = os.path.join(rel_dir, split)
            for file_name in os.listdir(rel_split_dir):
                epoch = int(file_name.split(".")[0])
                if epoch == 0 or epoch == best_epoch or epoch == 29:
                    continue
                # print(file_name)
                os.remove(os.path.join(rel_split_dir, file_name))