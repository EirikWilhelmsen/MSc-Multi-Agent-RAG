import matplotlib.pyplot as plt

file_path = "statistics.txt"

def create_graph(x_vals, y_vals, title, xlabel, ylabel):
    fig, ax = plt.subplots()

    bars = ax.bar(x_vals, y_vals)

    # Log-skala
    ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_vals)

    # Legg tall over hver søyle
    for bar, value in zip(bars, y_vals):
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height,                              
            f"{value}",                          
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.show()

def create_data(file_path):
    x_vals = []
    y_vals = []

    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # Skip "Article Counts:" header

        for line in lines:
            parts = line.strip().split()

            files_count = int(parts[0])  
            pageid_count = int(parts[2])  

            x_vals.append(files_count)
            y_vals.append(pageid_count)

    return x_vals, y_vals


if __name__ == "__main__":
    x, y = create_data(file_path)

    create_graph(
        x,
        y,
        "Distribution of Snapshots per Article",
        "Number of Snapshots per Article",
        "Number of Articles (PageIDs)"
    )