import os

class Paths:
    RAW_DATA_ROOT = "D:/Python/pattern-recognition-course-assignments/data"
    RGB_TRAIN_DIR = os.path.join(RAW_DATA_ROOT, "train_500", "rgb_data")
    DEPTH_TRAIN_DIR = os.path.join(RAW_DATA_ROOT, "train_500", "depth_data")
    INFRARED_TRAIN_DIR = os.path.join(RAW_DATA_ROOT, "train_500", "ir_data")
    TRAIN_LABELS_PATH = os.path.join(RAW_DATA_ROOT, "train_500", "train_videofolder_500.txt")

    RGB_TEST_DIR = os.path.join(RAW_DATA_ROOT, "test_200", "rgb_data")
    DEPTH_TEST_DIR = os.path.join(RAW_DATA_ROOT, "test_200", "depth_data")
    INFRARED_TEST_DIR = os.path.join(RAW_DATA_ROOT, "test_200", "ir_data")
    TEST_LABELS_PATH = os.path.join(RAW_DATA_ROOT, "test_200", "test_videofolder_200.txt")

    @classmethod
    def modify_root_dir(cls, new_root="/home/data/dataset"):
        cls.RAW_DATA_ROOT = new_root
        cls.RGB_TRAIN_DIR = os.path.join(new_root, "train_500", "rgb_data")
        cls.DEPTH_TRAIN_DIR = os.path.join(new_root, "train_500", "depth_data")
        cls.INFRARED_TRAIN_DIR = os.path.join(new_root, "train_500", "ir_data")
        cls.TRAIN_LABELS_PATH = os.path.join(new_root, "train_500", "train_videofolder_500.txt")

        cls.RGB_TEST_DIR = os.path.join(new_root, "test_200", "rgb_data")
        cls.DEPTH_TEST_DIR = os.path.join(new_root, "test_200", "depth_data")
        cls.INFRARED_TEST_DIR = os.path.join(new_root, "test_200", "ir_data")
        cls.TEST_LABELS_PATH = os.path.join(new_root, "test_200", "test_videofolder_200.txt")