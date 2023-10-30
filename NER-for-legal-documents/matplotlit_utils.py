def autolabel(ax, rects, threshold_for_rotation):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()

        if height < threshold_for_rotation:
            rotation = 90
        else:
            rotation = 0

        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=rotation)