# constants.py

OD_MODEL_FILES = {
    'yolov12s': {
        'url': 'local', 
        'filename': 'yolov12.pt'
    }, #yolo12
}

SEG_MODEL_FILES = { 
    'segmentation_model': {
        'url': 'local',
        'filename': 'segmentation_model.pt'
    },
}

OBJECT_DETECTION_CATEGORIES = {
    0: 'Bleb with mRNA',
    1: 'Oil Core',
    2: 'Other LNP',
    3: 'Not Fully Visible LNP',
    4: 'Multilamellar LNP',
    5: 'Multilamellar with mRNA',
}

OD_BLEB_WITH_MRNA = 0
OD_MRNA = 1
OD_OIL_CORE = 2
OD_OTHER_LNP = 3
OD_NOT_FULLY_VISIBLE_LNP = 4
OD_LIPOSOMAL_LNP = 5
OD_OIL_DROPLET = 6
OD_LIPOSOME = 7

OD_INNER_CATEGORIES = [OD_MRNA, OD_OIL_DROPLET]

SEG_BLEB_WITH_MRNA = 1
SEG_MRNA = 2
SEG_OIL_CORE = 3
SEG_OTHER_LNP = 4
SEG_NOT_FULLY_VISIBLE_LNP = 5
SEG_LIPOSOMAL_LNP = 6
SEG_OIL_DROPLET = 7
SEG_LIPOSOME = 8

SEG_INNER_CATEGORIES = [SEG_MRNA, SEG_OIL_DROPLET]

SEGMENTATION_CATEGORIES = {
    SEG_BLEB_WITH_MRNA: 'Bleb with mRNA',
    SEG_MRNA: 'mRNA',
    SEG_OIL_CORE: 'Oil Core',
    SEG_OTHER_LNP: 'Other LNP',
    SEG_NOT_FULLY_VISIBLE_LNP: 'Not Fully Visible LNP',
    SEG_LIPOSOMAL_LNP: 'Liposomal LNP',
    SEG_OIL_DROPLET: 'Oil Droplet',
    SEG_LIPOSOME: 'Liposome',
}

POST_PROCESS_BLEB_WITH_MRNA = 1
POST_PROCESS_MRNA = 2
POST_PROCESS_OIL_CORE = 3
POST_PROCESS_OTHER_LNP = 4
POST_PROCESS_NOT_FULLY_VISIBLE_LNP = 5
POST_PROCESS_LIPOSOMAL_LNP = 6
POST_PROCESS_OIL_DROPLET = 7
POST_PROCESS_LIPOSOME = 8
POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA = 9

POST_INNER_CATEGORIES = [POST_PROCESS_MRNA, POST_PROCESS_OIL_DROPLET]

POST_PROCESSING_CATEGORIES = {
    POST_PROCESS_BLEB_WITH_MRNA: 'Bleb with mRNA',
    POST_PROCESS_MRNA: 'mRNA',
    POST_PROCESS_OIL_CORE: 'Oil Core',
    POST_PROCESS_OTHER_LNP: 'Other LNP',
    POST_PROCESS_NOT_FULLY_VISIBLE_LNP: 'Not Fully Visible LNP',
    POST_PROCESS_LIPOSOMAL_LNP: 'Liposomal LNP',
    POST_PROCESS_OIL_DROPLET: 'Oil Droplet',
    POST_PROCESS_LIPOSOME: 'Liposome',
    POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA: 'Liposomal LNP with mRNA',
}

OD_TO_SEG_CATEGORIES_MAPPING = {
    OD_BLEB_WITH_MRNA: SEG_BLEB_WITH_MRNA,
    OD_MRNA: SEG_MRNA,
    OD_OIL_CORE: SEG_OIL_CORE,
    OD_OTHER_LNP: SEG_OTHER_LNP,
    OD_NOT_FULLY_VISIBLE_LNP: SEG_NOT_FULLY_VISIBLE_LNP,
    OD_LIPOSOMAL_LNP: SEG_LIPOSOMAL_LNP,
    OD_OIL_DROPLET: SEG_OIL_DROPLET,
    OD_LIPOSOME: SEG_LIPOSOME,
}

SEG_TO_OD_CATEGORIES_MAPPING = {
    SEG_BLEB_WITH_MRNA: OD_BLEB_WITH_MRNA,
    SEG_OIL_CORE: OD_OIL_CORE,
    # SEG_OTHER_LNP: OD_OTHER_LNP,
    SEG_NOT_FULLY_VISIBLE_LNP: OD_NOT_FULLY_VISIBLE_LNP,
    # SEG_MULTILAMELLAR_LNP: OD_MULTILAMELLAR_LNP,
}

SEG_TO_COCO_CATEGORIES_MAPPING = {
    SEG_BLEB_WITH_MRNA: 1,
    SEG_MRNA: 2,
    SEG_OIL_CORE: 3,
    SEG_OTHER_LNP: 4,
    SEG_NOT_FULLY_VISIBLE_LNP: 5,
    SEG_LIPOSOMAL_LNP: 6,
    SEG_OIL_DROPLET: 7,
    SEG_LIPOSOME: 8,
}

SEG_LABEL_COLORS = {
    SEG_BLEB_WITH_MRNA: '#FFCC33',
    SEG_MRNA: '#FFC6E5',
    SEG_OIL_CORE: '#66FF66',
    SEG_OTHER_LNP: '#33DDFF',
    SEG_NOT_FULLY_VISIBLE_LNP: '#FF355E',
    SEG_LIPOSOMAL_LNP: '#8C78F0',
    SEG_OIL_DROPLET: '#F59331',
    SEG_LIPOSOME: '#44bb99',
}

POST_PROCESS_LABEL_COLORS = {
    POST_PROCESS_BLEB_WITH_MRNA: '#FFCC33',
    POST_PROCESS_MRNA: '#FFC6E5',
    POST_PROCESS_OIL_CORE: '#66FF66',
    POST_PROCESS_OTHER_LNP: '#33DDFF',
    POST_PROCESS_NOT_FULLY_VISIBLE_LNP: '#FF355E',
    POST_PROCESS_LIPOSOMAL_LNP: '#8C78F0',
    POST_PROCESS_OIL_DROPLET: '#F59331',
    POST_PROCESS_LIPOSOME: '#44bb99',
    POST_PROCESS_LIPOSOMAL_LNP_WITH_MRNA: '#144c89',
}

PIXELS_IN_NM = 4.68 # nm
MRNA_SIZE = 208.44 # nm2
MRNA_VOLUME = 1999 # nm3
POLYGON_OVERLAP_THRESHOLD_PERC = 80 # percentage
LNP_AREA_THRESHOLD = 78.5 # nm2

SAM_MRNA_IOU_THRESHOLD = 0.3
SAM_MRNA_SIZE_RATIO = 0.8
SAM_MRNA_OVERLAP_THRESHOLD = 0.6
SAM_MRNA_CONTAINMENT_THRESHOLD = 0.8