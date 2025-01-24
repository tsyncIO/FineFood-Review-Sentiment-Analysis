import matplotlib.pyplot as plt

def visualize_predictions(images, labels, preds):
    """Visualize images with true and predicted labels"""
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(f"True: {labels[i]}, Pred: {preds[i]}")
        ax.axis('off')
    plt.show()
