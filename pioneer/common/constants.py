import numpy as np

NUMBERED_PICKLE_FMT = '{:08d}.pkl'
NUMBERED_PNG_FMT = '{:08d}.png'
NUMBERED_JPG_FMT = '{:08d}.jpg'

NUMBERED_PICKLE_PATTERN = r'([0-9]+)\.pkl'
NUMBERED_PNG_PATTERN = r'([0-9]+)\.png'
NUMBERED_JPG_PATTERN = r'([0-9]+)\.jpg'

TIMESTAMPS_CSV_PATTERN = 'timestamps.csv'
CONFIG_YML_PATTERN = 'config.yml'

PIXELL_HEAD_POSITIONS = {
    'left':   np.array([0.03479,  0.05655, 0.01562]),
    'center': np.array([0.03952,  0,       0.01562]),
    'right':  np.array([0.03479, -0.05655, 0.01562]),
}

ID_FROM_LCAS_CHAN_TO_SENSOR_CHAN = np.array([
        543,    542,    541,    540,    539,    538,    537,    536,    535,    534,    533,    532,    531,    530,    529,    528,    527,    526,    525,    524,    523,    522,    521,    520,    519,    518,    517,    516,    515,    514,    513,    512,

        287,    286,    285,    284,    283,    282,    281,    280,    279,    278,    277,    276,    275,    274,    273,    272,    271,    270,    269,    268,    267,    266,    265,    264,    263,    262,    261,    260,    259,    258,    257,    256,

        31,     30,     29,     28,     27,     26,     25,     24,     23,     22,     21,     20,     19,     18,     17,     16,     15,     14,     13,     12,     11,     10,     9,      8,      7,      6,      5,      4,      3,      2,      1,      0,

        575,    574,    573,    572,    571,    570,    569,    568,    567,    566,    565,    564,    563,    562,    561,    560,    559,    558,    557,    556,    555,    554,    553,    552,    551,    550,    549,    548,    547,    546,    545,    544,

        319,    318,    317,    316,    315,    314,    313,    312,    311,    310,    309,    308,    307,    306,    305,    304,    303,    302,    301,    300,    299,    298,    297,    296,    295,    294,    293,    292,    291,    290,    289,    288,

        63,     62,     61,     60,     59,     58,     57,     56,     55,     54,     53,     52,     51,     50,     49,     48,     47,     46,     45,     44,     43,     42,     41,     40,     39,     38,     37,     36,     35,     34,     33,     32,

        607,    606,    605,    604,    603,    602,    601,    600,    599,    598,    597,    596,    595,    594,    593,    592,    591,    590,    589,    588,    587,    586,    585,    584,    583,    582,    581,    580,    579,    578,    577,    576,

        351,    350,    349,    348,    347,    346,    345,    344,    343,    342,    341,    340,    339,    338,    337,    336,    335,    334,    333,    332,    331,    330,    329,    328,    327,    326,    325,    324,    323,    322,    321,    320,

        95,     94,     93,     92,     91,     90,     89,     88,     87,     86,     85,     84,     83,     82,     81,     80,     79,     78,     77,     76,     75,     74,     73,     72,     71,     70,     69,     68,     67,     66,     65,     64,

        639,    638,    637,    636,    635,    634,    633,    632,    631,    630,    629,    628,    627,    626,    625,    624,    623,    622,    621,    620,    619,    618,    617,    616,    615,    614,    613,    612,    611,    610,    609,    608,

        383,    382,    381,    380,    379,    378,    377,    376,    375,    374,    373,    372,    371,    370,    369,    368,    367,    366,    365,    364,    363,    362,    361,    360,    359,    358,    357,    356,    355,    354,    353,    352,

        127,    126,    125,    124,    123,    122,    121,    120,    119,    118,    117,    116,    115,    114,    113,    112,    111,    110,    109,    108,    107,    106,    105,    104,    103,    102,    101,    100,    99,     98,     97,     96,

        671,    670,    669,    668,    667,    666,    665,    664,    663,    662,    661,    660,    659,    658,    657,    656,    655,    654,    653,    652,    651,    650,    649,    648,    647,    646,    645,    644,    643,    642,    641,    640,

        415,    414,    413,    412,    411,    410,    409,    408,    407,    406,    405,    404,    403,    402,    401,    400,    399,    398,    397,    396,    395,    394,    393,    392,    391,    390,    389,    388,    387,    386,    385,    384,

        159,    158,    157,    156,    155,    154,    153,    152,    151,    150,    149,    148,    147,    146,    145,    144,    143,    142,    141,    140,    139,    138,    137,    136,    135,    134,    133,    132,    131,    130,    129,    128,

        703,    702,    701,    700,    699,    698,    697,    696,    695,    694,    693,    692,    691,    690,    689,    688,    687,    686,    685,    684,    683,    682,    681,    680,    679,    678,    677,    676,    675,    674,    673,    672,

        447,    446,    445,    444,    443,    442,    441,    440,    439,    438,    437,    436,    435,    434,    433,    432,    431,    430,    429,    428,    427,    426,    425,    424,    423,    422,    421,    420,    419,    418,    417,    416,

        191,    190,    189,    188,    187,    186,    185,    184,    183,    182,    181,    180,    179,    178,    177,    176,    175,    174,    173,    172,    171,    170,    169,    168,    167,    166,    165,    164,    163,    162,    161,    160,

        735,    734,    733,    732,    731,    730,    729,    728,    727,    726,    725,    724,    723,    722,    721,    720,    719,    718,    717,    716,    715,    714,    713,    712,    711,    710,    709,    708,    707,    706,    705,    704,

        479,    478,    477,    476,    475,    474,    473,    472,    471,    470,    469,    468,    467,    466,    465,    464,    463,    462,    461,    460,    459,    458,    457,    456,    455,    454,    453,    452,    451,    450,    449,    448,

        223,    222,    221,    220,    219,    218,    217,    216,    215,    214,    213,    212,    211,    210,    209,    208,    207,    206,    205,    204,    203,    202,    201,    200,    199,    198,    197,    196,    195,    194,    193,    192,

        767,    766,    765,    764,    763,    762,    761,    760,    759,    758,    757,    756,    755,    754,    753,    752,    751,    750,    749,    748,    747,    746,    745,    744,    743,    742,    741,    740,    739,    738,    737,    736,

        511,    510,    509,    508,    507,    506,    505,    504,    503,    502,    501,    500,    499,    498,    497,    496,    495,    494,    493,    492,    491,    490,    489,    488,    487,    486,    485,    484,    483,    482,    481,    480,

        255,    254,    253,    252,    251,    250,    249,    248,    247,    246,    245,    244,    243,    242,    241,    240,    239,    238,    237,    236,    235,    234,    233,    232,    231,    230,    229,    228,    227,    226,    225,    224
    ])


PIXELL_FILTERS_KERNEL = {
    'low': np.array([0.000703, 0.003722, 0.011017, 0.021974, 0.034887, 0.047587, 0.059213, 0.069354, 
                    0.076891, 0.081912, 0.084969, 0.086764, 0.084437, 0.080814, 0.073762, 0.062209,
                    0.047612, 0.033038, 0.019663, 0.010658, 0.005447, 0.002853, 0.001615, 0, 0]),
    'high': np.array([0.004976, 0.010496, 0.018112, 0.027080, 0.037099, 0.047171, 0.056302, 0.064230, 0.070783, 
                    0.075466, 0.078202, 0.080508, 0.077462, 0.073063, 0.065948, 0.056636, 0.045911, 0.035610,
                    0.026766, 0.019581, 0.013840, 0.009585, 0.006684, 0, 0])
}
