from experiment1 import *
from experiment2 import *

if __name__ == '__main__':
    random_seed_global(0)

    dataset_augmented_path = Path('dataset1-augmented')
    #dataset_augmented_path = Path('dataset2-augmented')
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
    #run_baseline_2(epochs=20,)
    epochs = 50
    #run_char_cnn_2(epochs=4, get_test=True)
    #run_word_cnn_2(epochs=17, get_test=True)
    preprocessing = [0]
    # run_char_cnn_2(preprocessing, epochs=epochs, get_test=False)
    # run_word_cnn_2(preprocessing, epochs=epochs, get_test=False)
    preprocessing = [1]
    #run_char_cnn_2(preprocessing, epochs=13, get_test=True)
    # run_word_cnn_2(preprocessing, epochs=epochs, get_test=False)
    preprocessing = [2]
    #run_char_cnn_2(preprocessing, epochs=11, get_test=True)
    # run_word_cnn_2(preprocessing, epochs=epochs, get_test=False)
    #
    #run_char_cnn_2(augmentation=augment_sr, epochs=19, get_test=False)
    #run_word_cnn_2(augmentation=augment_sr, epochs=48, get_test=True)
    #
    #run_char_cnn_2(augmentation=augment_ri, epochs=7, get_test=True)
    #run_word_cnn_2(augmentation=augment_ri, epochs=9, get_test=True)
    #
    #run_char_cnn_2(augmentation=augment_rs, epochs=7, get_test=False)
    #run_word_cnn_2(augmentation=augment_rs, epochs=7, get_test=True)
    #
    #run_char_cnn_2(augmentation=augment_rd, epochs=9, get_test=True)
    #run_word_cnn_2(augmentation=augment_rd, epochs=7, get_test=True)
