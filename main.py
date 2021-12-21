from experiment1 import *
from experiment3 import *
from experiment_sentiment import *

if __name__ == '__main__':
    random_seed_global(0)

    dataset_augmented_path = Path('dataset1-augmented')
    augment_sr = {
        "percentage_sr": 0.1,
        "num_aug": 2,
        "target_path": dataset_augmented_path / 'sr.npz',
    }
    augment_ri = {
        "percentage_ri": 0.1,
        "num_aug": 2,
        "target_path": dataset_augmented_path / 'ri.npz',
    }
    augment_rs = {
        "percentage_rs": 0.1,
        "num_aug": 2,
        "target_path": dataset_augmented_path / 'rs.npz',
    }
    augment_rd = {
        "percentage_rd": 0.1,
        "num_aug": 2,
        "target_path": dataset_augmented_path / 'rd.npz',
    }
    run_baseline_1(epochs=10,)
    preprocessing = [0]
    run_char_cnn_1(preprocessing, epochs=50, get_test=False)
    run_word_cnn_1(preprocessing, epochs=50, get_test=False)
    preprocessing = [1]
    run_char_cnn_1(preprocessing, epochs=50, get_test=False)
    run_word_cnn_1(preprocessing, epochs=50, get_test=False)
    preprocessing = [2]
    run_char_cnn_1(preprocessing, epochs=50, get_test=False)
    run_word_cnn_1(preprocessing, epochs=50, get_test=False)

    run_char_cnn_1(augmentation=augment_sr, epochs=50, get_test=False)
    run_word_cnn_1(augmentation=augment_sr, epochs=50, get_test=False)

    run_char_cnn_1(augmentation=augment_ri, epochs=50, get_test=False)
    run_word_cnn_1(augmentation=augment_ri, epochs=50, get_test=False)

    run_char_cnn_1(augmentation=augment_rs, epochs=50, get_test=False)
    run_word_cnn_1(augmentation=augment_rs, epochs=50, get_test=False)

    run_char_cnn_1(augmentation=augment_rd, epochs=50, get_test=False)
    run_word_cnn_1(augmentation=augment_rd, epochs=50, get_test=False)
