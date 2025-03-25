import matplotlib.pyplot as plt
import random

def plot_coplanar_lines(lines, labels, image):
    """
    Visualize the lines on an image with colors corresponding to their plane labels.
    Outliers (label -1) are drawn in grey.
    """
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    print(f"Unique Labels: {unique_labels}")
    print(f"Number of Clusters: {num_clusters}")

    # Generate random colors for clusters.
    random.seed(42)  # For reproducibility.
    colors = [tuple(random.random() for _ in range(3)) for _ in range(num_clusters)]
    random.shuffle(colors)
    label_to_color = {label: colors[idx] for idx, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    for idx, line in enumerate(lines):
        label = labels[idx]
        color = 'grey' if label == -1 else label_to_color.get(label, (0, 0, 0))
        plt.plot(
            [line[0, 0], line[1, 0]], 
            [line[0, 1], line[1, 1]], 
            color=color, 
            linewidth=2
        )
    plt.axis("off")
    plt.show()
    
def plot_lines_confidence(img, lines, confidences, indices=(0, 1), alpha=1):
    """ Plots lines with color intensity based on confidence scores """
    fig, ax = plt.subplots()
    #norm = plt.Normalize(min(confidences), max(confidences))
    cmap = plt.cm.get_cmap("coolwarm")
    
    for i, l in enumerate(lines):
        color = cmap((confidences[i]))
        line = plt.Line2D((l[0, 0], l[1, 0]), (l[0, 1], l[1, 1]), linewidth=2, color=color, alpha=alpha)
        ax.add_line(line)
    
    ax.imshow(img, cmap='gray')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label='Confidence Score')
    plt.show()