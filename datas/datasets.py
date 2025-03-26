import os
import scipy.io as sio

from typing import Tuple

from datas.base import HSIDataset


class HyRankDataset(HSIDataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(HyRankDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)
        if split == 'train':
            data_filename = 'Dioni.mat'
            gt_filename = 'Dioni_gt.mat'
        else:
            # 验证集等于测试集
            data_filename = 'Loukia.mat'
            gt_filename = 'Loukia_gt.mat'
        self.data_path = os.path.join(root, data_filename)
        self.data = sio.loadmat(self.data_path)['ori_data'].astype('float32')
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.gt_path)['map'].astype('int')
        self.gt_raw = self.gt.copy()

        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    def selector(self, x, y):
        return y not in [0, 6, 8]

    @property
    def num_channels(self):
        return 176

    @property
    def names(self):
        return [
            'Dense urban fabric',
            'Mineral extraction sites',
            'Non-irrigated arable land',
            'Fruit trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense scleroph, Vegetation',
            'Sparse scleroph, Vegetation',
            'Sparse vegetated areas',
            'Rocks & sand',
            'Water',
            'Coastal Water'
        ]

    @property
    def class_prompts(self):
        return [
            'Dense urban fabric with a high concentration of buildings, roads, and other infrastructure, often exhibiting a compact and continuous structure. This category is characterized by the presence of densely packed urban features, such as multi-story buildings, paved roads, and limited green spaces.',
            'Areas where minerals are being actively being mined, including distinct spectral signatures that correspond to different types of mineralized rock, waste piles, open-pit mines with mining activity.',
            'Non-irrigated land where crops or vegetation grow without supplemental irrigation, relying solely on natural rainfall for water. This land may include drylands, rainfed agricultural fields, or natural vegetation in semi-arid or arid regions.',
            'Images of various fruit tree species provides detailed information on the health, growth, and characteristics of fruit trees, such as apple, orange, and pear trees, among others.',
            'Olive groves with healthy foliage, fruit, and sometimes the soil types in olive groves.',
            'Coniferous forest, mainly dense, evergreen tree cover dominated by conifer species such as pines, firs, and spruces.',
            'Dense vegetation with lush, healthy plant life, such as forests, grasslands, or agricultural crops.',
            'Sparce vegetation, where vegetation is present but not densely packed. These regions may include scattered or patchy vegetation cover, such as grasslands, sparse shrublands, or areas with low vegetation density due to environmental factors like drought, soil quality, or human activity.',
            'Sparce areas that correspond to homogeneous or unvaried materials such as bare soil and shadowed regions.',
            'Image of different rocks and sand types.',
            'Water areas such as rivers, lakes, and ponds, typically characterized by clearer water conditions that are less influenced by land-based factors.',
            'Coastal waters such as estuaries and lagoons, which are typically influenced by proximity to land, such as shallow coastal areas, estuaries, and nearshore zones. These waters often exhibit increased turbidity, higher concentrations of suspended sediments, and various dissolved substances.'
        ]

    @property
    def domain_prompts(self):
        return [
            'Hyperspectral image captured in Greece, a diverse landscape with varied terrain and vegetation types.',
            'Hyperspectral image captured in Greece, a more homogeneous environment. The uniformity in vegetation and terrain simplifies.'
        ]

    @property
    def pixels(self):
        return [
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105],
            [252, 205, 229],
            [217, 217, 217],
            [188, 128, 189],
            [204, 128, 189],
            [255, 237, 111]
        ]


class ShangHangDataset(HSIDataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(ShangHangDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)

        data_filename = 'DataCube_ShanghaiHangzhou.mat'
        self.data_path = os.path.join(root, data_filename)
        raw = sio.loadmat(self.data_path)
        if split == 'train':
            self.data = raw['DataCube2'].astype('float32')
            self.gt = raw['gt2'].astype('int')
        else:
            self.data = raw['DataCube1'].astype('float32')
            self.gt = raw['gt1'].astype('int')
        self.gt_raw = self.gt.copy()
        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    @property
    def num_channels(self):
        return 198

    @property
    def names(self):
        return [
            'Water',
            'Land/Building',
            'Plant'
        ]

    @property
    def class_prompts(self):
        return [
            'Water areas such as rivers, lakes, and ponds, typically characterized by clearer water conditions that are less influenced by land-based factors.',
            'Urban structures and terrestrial surfaces in the region, including roads, residential and commercial buildings, and other man-made infrastructures.',
            'Image of a variety of vegetation types, including crops, trees, and grasses.'
        ]

    @property
    def domain_prompts(self):
        return [
            "Hyperspectral image of Hangzhou, a blend of modern development and natural landscapes. Hangzhou's urban areas are interspersed with parks, waterways, and historical sites, contributing to a more varied and less densely built environment.",
            "Hyperspectral image captured in Shanghai, characterized by its modern skyline, featuring numerous high-rise buildings, skyscrapers, and expansive urban areas. The urban environment is often marked by a high density of buildings and a significant presence of commercial and residential complexes."
        ]

    @property
    def pixels(self):
        return [
            [0, 157, 130],
            [255, 255, 85],
            [110, 101, 172]
        ]
