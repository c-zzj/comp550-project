from experiment1 import *
from experiment3 import *

if __name__ == '__main__':
    random_seed_global(0)
    #run_baseline_2()
    dataset1_augmented_path = Path('dataset1-augmented')
    dataset3_augmented_path = Path('dataset3-augmented')
    augment_sr = {
        "percentage_sr": 0.1,
        "num_aug": 2,
        "target_path": dataset1_augmented_path / 'sr.npz',
    }
    augment_ri = {
        "percentage_ri": 0.1,
        "num_aug": 2,
        "target_path": dataset1_augmented_path / 'ri.npz',
    }
    augment_rs = {
        "percentage_rs": 0.1,
        "num_aug": 2,
        "target_path": dataset1_augmented_path / 'rs.npz',
    }
    augment_rd = {
        "percentage_rd": 0.1,
        "num_aug": 2,
        "target_path": dataset1_augmented_path / 'rd.npz',
    }
    preprocessing = [2]
    #run_baseline_1(epochs=40,)
    run_char_cnn_1(augmentation=augment_sr, epochs=20, get_test=False)
    #run_word_cnn_1(preprocessing, epochs=9, get_test=True)
