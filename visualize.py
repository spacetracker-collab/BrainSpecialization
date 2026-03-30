
import matplotlib.pyplot as plt

def plot_history(history):
    loss = [h[0] for h in history]
    shortcut = [h[1] for h in history]
    cognitive = [h[2] for h in history]

    plt.figure()
    plt.plot(loss, label="Loss")
    plt.plot(shortcut, label="Shortcut strength")
    plt.plot(cognitive, label="Cognitive strength")
    plt.legend()
    plt.show()
