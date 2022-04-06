import itertools
import os.path as osp
import numpy as np

from PIL import Image


def product(*inputs):
    return [''.join(e) for e in itertools.product(*inputs)]

dataset_classes = {
    'DOTA1_0': ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor'),
    'DOTA1_5': ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane'),
    'DOTA2_0': ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'airport', 'helipad'),
    'DIOR': ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'expressway-service-area', 'expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
             'windmill'),
    'HRSC': ('ship', ),
    'HRSC_cls': ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                 '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                 '31', '32', '33'),
    'MSRA_TD500': ('text', ),
    'RCTW_17': ('text', ),
    'VOC': ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
            'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle',
            'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'),
}

dataset_aliases = {
    'DOTA1_0': product(['dota', 'DOTA'], ['', '1', '1.0', '1_0']),
    'DOTA1_5': product(['dota', 'DOTA'], ['1.5', '1_5']),
    'DOTA2_0': product(['dota', 'DOTA'], ['2', '2.0', '2_0']),
    'DIOR': ['dior', 'DIOR'],
    'HRSC': product(['hrsc', 'HRSC'], ['', '2016']),
    'HRSC_cls': product(['hrsc', 'HRSC'], ['_cls', '2016_cls']),
    'MSRA_TD500': ['msra_td500', 'MSRA_TD500', 'msra-td500', 'MSRA-TD500'],
    'RCTW_17': ['rctw_17', 'RCTW_17', 'rctw-17', 'RCTW-17'],
    'VOC': ['VOC', 'voc'],
}

img_exts = ['.jpg', '.JPG', '.png', '.tif', '.bmp']


def read_img_info(imgpath):
    imgfile = osp.split(imgpath)[-1]
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    size = Image.open(imgpath).size
    content = dict(width=size[0], height=size[1], filename=imgfile, id=img_id)
    return content


def get_classes(alias_or_list):
    if isinstance(alias_or_list, str):
        if osp.isfile(alias_or_list):
            class_names = []
            with open(alias_or_list) as f:
                for line in f:
                    class_names.append(line.strip())
            return tuple(class_names)

        for k, v in dataset_aliases.items():
            if alias_or_list in v:
                return dataset_classes[k]

        return alias_or_list.split('|')

    if isinstance(alias_or_list, (list, tuple)):
        classes = []
        for item in alias_or_list:
            for k, v in dataset_aliases.items():
                if item in v:
                    classes.extend(dataset_classes[k])
                    break
            else:
                classes.append(item)
        return tuple(classes)

    raise TypeError(
        f'input must be a str, list or tuple but got {type(alias_or_list)}')


def change_cls_order(contents, old_classes, new_classes):
    for n_c, o_c in zip(new_classes, old_classes):
        if n_c != o_c:
            break
    else:
        if len(old_classes) == len(new_classes):
            return

    new_cls2lbl = {cls: i for i, cls in enumerate(new_classes)}
    lbl_mapper = [new_cls2lbl[cls] if cls in new_cls2lbl else -1
                  for cls in old_classes]
    lbl_mapper = np.array(lbl_mapper, dtype=np.int64)

    for content in contents:
        new_labels = lbl_mapper[content['ann']['labels']]
        if (new_labels == -1).any():
            inds = np.nonzero(new_labels != -1)[0]
            for k, v in content['ann'].items():
                try:
                    content['ann'][k] = v[inds]
                except TypeError:
                    content['ann'][k] = [v[i] for i in inds]
        else:
            content['ann']['labels'] = new_labels


def merge_prior_contents(bases, priors, merge_type='addition'):
    id_mapper = {base['id']: i for i, base in enumerate(bases)}
    for prior in priors:
        img_id = prior['id']
        if img_id not in id_mapper:
            continue

        base = bases[id_mapper[img_id]]
        for key in prior.keys():
            if key in ['id', 'filename', 'width', 'height', 'ann']:
                continue
            if (key not in base) or (base[key] is None) or (merge_type == 'replace'):
                base[key] = prior[key]

        if 'ann' in prior:
            if not base.get('ann', {}):
                base['ann'] = prior['ann']
            else:
                base_anns, prior_anns = base['ann'], prior['ann']
                assert base_anns.keys() == prior_anns.keys()
                for key in prior_anns:
                    if isinstance(base_anns[key], np.ndarray):
                        base_anns[key] = prior_anns[key] if merge_type == 'replace' \
                                else np.concatenate([base_anns[key], prior_anns[key]], axis=0)
                    elif isinstance(base_anns[key], list):
                        base_anns[key] = prior_anns[key] if merge_type == 'replace' \
                                else base_anns[key].update(prior_anns[key])
                    else:
                        raise TypeError("annotations only support np.ndarrya and list"+
                                        f", but get {type(base_anns[key])}")


def split_imgset(contents, imgset):
    id_mapper = {content['id']: i for i, content in enumerate(contents)}
    assert isinstance(imgset, (list, tuple, str))
    if isinstance(imgset, str):
        with open(imgset, 'r') as f:
            imgset = [line for line in f]

    imgset_contents = []
    for img_id in imgset:
        img_id = osp.split(img_id.strip())[-1]
        img_id = osp.splitext(img_id)[0]
        if img_id not in id_mapper:
            print(f"Can't find ID:{img_id} image!")
            continue

        imgset_contents.append(contents[id_mapper[img_id]])
    return imgset_contents


class _ConstMapper:

    def __init__(self, const_value):
        self.const_value = const_value

    def __getitem__(self, key):
        return self.const_value

    def __contains__(self, key):
        return True
