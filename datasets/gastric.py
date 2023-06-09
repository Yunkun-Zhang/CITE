import os.path as osp
import numpy as np
import random
import copy
from mmcls.datasets import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy


@DATASETS.register_module()
class PatchGastricCls3(BaseDataset):
    CLASSES = [str(i) for i in range(3)]

    def __init__(self,
                 patient_level=False,
                 slide_balance=False,
                 patch_balance=False,
                 feature_file=None,
                 num_aug=None,
                 **kwargs):
        self.patient_level = patient_level
        assert not (slide_balance and patch_balance)
        self.slide_balance = slide_balance
        self.patch_balance = patch_balance
        self.feature_file = feature_file
        self.num_aug = num_aug
        super().__init__(**kwargs)

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        if self.feature_file is not None:
            assert self.num_aug >= 1
            features = []
            for a in range(self.num_aug):
                feature = np.load(self.feature_file + f'_{a + 1}.npy')
                features.append(feature)
            features = np.array(features)

        data_infos = []
        with open(self.ann_file, 'r') as f:
            for i, line in enumerate(f):
                filename, class_name = line.strip().split(' ')
                gt_label = int(class_name)  # class_to_idx
                bag = filename.split('_')[0]
                info = {
                    'img_prefix': None,
                    'img_info': {
                        'filename': osp.join(self.data_prefix, filename)
                    },
                    'gt_label': np.array(gt_label, dtype=np.int64),
                    'bag': bag,
                    'img_text': ''
                }
                if self.feature_file is not None:
                    info['feature'] = features[:, i]
                data_infos.append(info)

        self.class_to_slide = [set() for _ in range(len(self.CLASSES))]
        self.patient_infos = dict()
        for i, info in enumerate(data_infos):
            patient_id = info['bag']
            patient_class = info['gt_label'].item()
            self.class_to_slide[patient_class].add(patient_id)
            self.patient_infos[patient_id] = self.patient_infos.get(patient_id, []) + [i]
        self.class_to_slide = [list(x) for x in self.class_to_slide]

        self.class_counter = 0
        if self.slide_balance:
            self.slide_counter = [0 for _ in range(len(self.CLASSES))]
            self.patch_counter = [[0 for _ in self.class_to_slide[c]] for c in range(len(self.CLASSES))]
        elif self.patch_balance:
            self.patch_counter = [0 for _ in range(len(self.CLASSES))]
            self.class_to_patch = [sum([self.patient_infos[i] for i in self.class_to_slide[c]], [])
                                   for c in range(len(self.CLASSES))]
            for c in range(len(self.CLASSES)):
                random.shuffle(self.class_to_patch[c])

        return data_infos

    def __getitem__(self, idx):
        if not (self.slide_balance or self.patch_balance):
            return self.prepare_data(idx)
        c = self.class_counter
        if self.slide_balance:
            s = self.slide_counter[c]
            p = self.patch_counter[c][s]
            patient_id = self.class_to_slide[c][s]
            data_idx = self.patient_infos[patient_id][p]
            results = copy.deepcopy(self.data_infos[data_idx])
            self.patch_counter[c][s] = (p + 1) % len(self.patient_infos[patient_id])
            self.slide_counter[c] = (s + 1) % len(self.class_to_slide[c])
            if self.patch_counter[c][s] == 0:
                random.shuffle(self.patient_infos[patient_id])
            if self.slide_counter[c] == 0:
                slide_idx = list(range(len(self.class_to_slide[c])))
                random.shuffle(slide_idx)
                self.patch_counter[c] = [self.patch_counter[c][s] for s in slide_idx]
                self.class_to_slide[c] = [self.class_to_slide[c][s] for s in slide_idx]
        elif self.patch_balance:
            p = self.patch_counter[c]
            data_idx = self.class_to_patch[c][p]
            results = copy.deepcopy(self.data_infos[data_idx])
            self.patch_counter[c] = (p + 1) % len(self.class_to_patch[c])
            if self.patch_counter[c] == 0:
                random.shuffle(self.class_to_patch[c])
        self.class_counter = (c + 1) % len(self.CLASSES)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mean_bag_accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        if metric_options is None:
            metric_options = dict()
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'class_accuracy', 'mean_accuracy',
            'bag_accuracy', 'bag_class_accuracy', 'mean_bag_accuracy'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        agg_hard = metric_options.get('agg_hard', False)
        if 'bag_accuracy' in metrics or 'mean_bag_accuracy' in metrics or 'bag_class_accuracy' in metrics:
            patient_ids = list(self.patient_infos.keys())
            bag_gt_labels = np.array([gt_labels[self.patient_infos[p][0]] for p in patient_ids])
            if agg_hard:
                raise NotImplementedError
            else:
                # soft voting
                bag_results = np.array([np.mean(results[self.patient_infos[p]], axis=0) for p in patient_ids])
            if 'bag_accuracy' in metrics:
                acc = accuracy(bag_results, bag_gt_labels, topk=1)
                eval_results.update({'bag_accuracy': acc.item()})
            if 'mean_bag_accuracy' in metrics or 'bag_class_accuracy' in metrics:
                mean_acc = 0.
                for i in range(len(self.CLASSES)):
                    idx = bag_gt_labels == i
                    acc = accuracy(bag_results[idx], bag_gt_labels[idx], topk=1)
                    mean_acc += acc.item()
                    eval_results.update({f'bag_accuracy_{i}': acc.item()})
                eval_results.update({'mean_bag_accuracy': mean_acc / len(self.CLASSES)})

        if 'accuracy' in metrics:
            acc = accuracy(results, gt_labels, topk=1)
            eval_results.update({'accuracy': acc.item()})

        if 'mean_accuracy' in metrics or 'class_accuracy' in metrics:
            mean_acc = 0.
            for i in range(len(self.CLASSES)):
                idx = gt_labels == i
                acc = accuracy(results[idx], gt_labels[idx], topk=1)
                mean_acc += acc.item()
                eval_results.update({f'accuracy_{i}': acc.item()})
            eval_results.update({'mean_accuracy': mean_acc / len(self.CLASSES)})

        return eval_results
