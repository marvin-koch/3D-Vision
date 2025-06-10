import matplotlib.pyplot as plt
import random
import numpy as np

def plot_images(images, titles, cmaps=None):
    num = len(images)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, num, i + 1)
        if cmaps is not None:
            cmap = cmaps if isinstance(cmaps, str) else cmaps[i]
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
    
    
def plot_lines_bool(ax, img, lines, is_correct, alpha=1):
    """Plot lines on an axis with blue for True and red for False."""
    colors = ['red' if not c else 'blue' for c in is_correct]
    
    for i, l in enumerate(lines):
        line = plt.Line2D(
            (l[0, 0], l[1, 0]),
            (l[0, 1], l[1, 1]),
            linewidth=2,
            color=colors[i],
            alpha=alpha
        )
        ax.add_line(line)
    
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()



def plot_coplanar_lines(ax, lines, labels, image):
    """
    Visualize lines on an image with colors corresponding to their plane labels.
    Outliers (label -1) are drawn in grey. Designed to be used with a subplot axis.
    """
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    # Generate random colors for clusters (excluding -1 if present)
    random.seed(42)
    colors = [tuple(random.random() for _ in range(3)) for _ in range(num_clusters)]
    random.shuffle(colors)
    label_to_color = {label: colors[idx] for idx, label in enumerate(unique_labels)}

    ax.imshow(image)
    for idx, line in enumerate(lines):
        label = labels[idx]
        color = 'grey' if label == -1 or label == 0 else label_to_color.get(label, (0, 0, 0))
        ax.plot(
            [line[0, 0], line[1, 0]],
            [line[0, 1], line[1, 1]],
            color=color,
            linewidth=2
        )

    ax.set_title("Coplanar Lines")
    ax.axis('off')
    
    
def visualize_plane_clusters(labels_2d, title):
    """
    Visualize plane cluster labels as a color-coded image.
    labels_2d: (H, W) integer labels.
    """
    plt.figure()
    unique_labels = np.unique(labels_2d)
    num_labels = len(unique_labels)

    # Generate random colors for each cluster
    # shape (num_labels, 3)
    colors = np.random.rand(num_labels, 3)

    # Build color image
    H, W = labels_2d.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        color_img[labels_2d == label] = colors[i]

    plt.imshow(color_img)
    plt.title(f"Plane Clusters  {title}")
    plt.axis("off")
    plt.show()
    
def color_map(label_map, num_labels):
    cols = np.random.randint(0,255,(num_labels,3),np.uint8)
    im = cols[label_map]
    im[label_map==0] = 0
    return im

 