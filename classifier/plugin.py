from classifier import *
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def SaveModel(folder_path: Path, save_last: bool = False, step: int = 1) -> TrainingPlugin:
    """
    :param folder_path: the path of the folder to save the model
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves the model after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            torch.save(clf.network.state_dict(), str(folder_path / f"{epoch}.params"))
            if save_last:
                if Path(folder_path / f"{epoch - 1}.params").exists():
                    Path.unlink(Path(folder_path / f"{epoch - 1}.params"))
    return plugin


def SaveGoodModels(folder_path: Path, metric: Metric, tolerance=0.01, step: int = 1) -> TrainingPlugin:
    """
    :param folder_path: the path of the folder to save the model
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves the model after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if epoch != 1:
                e = clf._tmp['learning_path']['epochs']
                val = clf._tmp['learning_path'][str(metric)]['val']
                indices_to_remove = []
                best = max(val)
                for i in range(len(val)):
                    if abs(val[i] - best) >= tolerance:
                        indices_to_remove.append(i)

                to_remove = [e[j] for j in indices_to_remove]
                for k in to_remove:
                    if Path(folder_path / f"{k}.params").exists():
                        Path.unlink(Path(folder_path / f"{k}.params"))

            torch.save(clf.network.state_dict(), str(folder_path / f"{epoch}.params"))

    return plugin


def SaveTrainingMessage(folder_path: Path, step: int = 1, empty_previous: bool = True) -> TrainingPlugin:
    """
    :param empty_previous:
    :param folder_path: the path of the log to be saved
    :param step: step size of epochs to activate the plugin
    :return: a plugin that appends the training message to the log file after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            with open(str(folder_path / 'log.txt'), 'a+') as f:
                f.write(clf.training_message)
    return plugin


def ElapsedTime(print_to_console: bool = True, log: bool = True, step: int = 1) -> TrainingPlugin:
    """

    :param print_to_console:
    :param log:
    :param step:
    :return:
    """
    start_time = timer()

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if 'time' in clf._tmp:
                s = f"time elapsed: {timer() - clf._tmp['time']} sec\n"
            else:
                s = f"time elapsed: {timer() - start_time} sec\n"
            if print_to_console:
                print(s, end='')
            if log:
                clf.training_message += s
            clf._tmp['time'] = timer()
    return plugin


def CalcTrainValPerformance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves training and validation performance to the temporary variable
    """

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            clf._tmp[str(metric)] = (
                clf.train_performance(metric, batch_size),
                clf.val_performance(metric, batch_size)
            )
    return plugin


def LogTrainValPerformance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that logs training and validation performance to training message after each step
    """
    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if str(metric) not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp[str(metric)][0], clf._tmp[str(metric)][1]
            s = f"TRAIN: {train}\tVAL: {val}, METRIC: {str(metric)}\n"
            clf.training_message += s
    return plugin


def PrintTrainValPerformance(metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that prints training and validation performance after each step
    """

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if str(metric) not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp[str(metric)]
            s = f"TRAIN: {train}\tVAL: {val}, METRIC: {str(metric)}"
            print(s)
    return plugin


def SaveTrainValPerformance(folder_path: Path, metric: Metric, batch_size: int = 300, step: int = 1) -> TrainingPlugin:
    """

    :param folder_path:
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves training and validation performance after each step
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if str(metric) not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp[str(metric)]
            if 'learning_path' not in clf._tmp:
                clf._tmp['learning_path'] = {'epochs': []}
            if str(metric) not in clf._tmp['learning_path']:
                clf._tmp['learning_path'][str(metric)] = {'train': [], 'val': []}
            clf._tmp['learning_path']['epochs'].append(epoch)
            clf._tmp['learning_path'][str(metric)]['train'].append(train)
            clf._tmp['learning_path'][str(metric)]['val'].append(val)
            torch.save(clf._tmp['learning_path'], str(folder_path / LEARNING_PATH_FNAME))
    return plugin


def load_train_val_performance(folder_path: Path) -> Dict[str, list]:
    """
    load the saved learning path generated by SaveTrainValPerformance
    :param folder_path:
    :return: {'epochs': List[float], metric: {'train': List[float], 'val': List[float]} }
    """
    if Path(folder_path / LEARNING_PATH_FNAME).exists():
        return torch.load(folder_path / LEARNING_PATH_FNAME)
    else:
        raise FileNotFoundError(f"The learning path for {folder_path} has not been saved!")


def PlotTrainValPerformance(folder_path: Path,
                            title: str,
                            metric: Metric,
                            show: bool = True,
                            save: bool = False,
                            batch_size: int = 300,
                            step: int = 1) -> TrainingPlugin:
    """

    :param save:
    :param title:
    :param folder_path:
    :param metric:
    :param batch_size: size of the batch of each prediction (for solving the GPU out-of-memory problem)
    :param step: step size of epochs to activate the plugin
    :return: a plugin that saves and/or shows the learning curve
    """
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    epochs = []
    train_performances = []
    val_performances = []

    def plugin(clf: NNClassifier, epoch: int) -> None:
        if epoch % step == 0:
            if str(metric) not in clf._tmp:
                train, val = clf.train_performance(metric, batch_size), clf.val_performance(metric, batch_size)
            else:
                train, val = clf._tmp[str(metric)]
            epochs.append(epoch)
            train_performances.append(train)
            val_performances.append(val)
            plt.figure()
            plt.plot(epochs, train_performances,
                     label="training", alpha=0.5)
            plt.plot(epochs, val_performances,
                     label="validation", alpha=0.5)
            # plt.ylim(top=1, bottom=0.9)
            plt.xlabel('Number of epochs')
            plt.ylabel(str(metric))
            plt.title(title)
            plt.legend()
            if save:
                plt.savefig(folder_path / f'{epoch} epochs.jpg')
                # delete previous plot
                if Path(folder_path / f'{epoch - step} epochs.jpg').exists():
                    Path.unlink(Path(folder_path / f'{epoch - step} epochs.jpg'))
            if show:
                plt.show()
            plt.close('all')
    return plugin
