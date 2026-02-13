from matplotlib.colors import LinearSegmentedColormap


custom_colors = [
    "#00A878", # greenish
    "#E6D98C", # yellowish
    "#CC2222" # redish
]

# Create a ListedColormap
green_yellow_red = LinearSegmentedColormap.from_list("custom_gradient", custom_colors)
