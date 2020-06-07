import matplotlib.pyplot as plt


def plot_history(history, *, y_lab):
    plt.plot(history['train'], '-', label='Train')
    plt.plot(history['val'], '-', label='Val')
    plt.xlabel("Epoch")
    plt.ylabel(y_lab)
    plt.legend()
    plt.show()