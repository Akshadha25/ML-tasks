
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

gmm_labels = gmm.fit_predict(X_scaled)

hierarchical = AgglomerativeClustering(
    n_clusters=3
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

result_data = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Gaussian Mixture",
        "Hierarchical"
    ],
    "Clusters": [
        3,
        3,
        3
    ]
})

kmeans_count = pd.Series(kmeans_labels).value_counts().sort_index()
gmm_count = pd.Series(gmm_labels).value_counts().sort_index()
hierarchical_count = pd.Series(hierarchical_labels).value_counts().sort_index()

cluster_count_data = pd.DataFrame({
    "Cluster": [1, 2, 3],
    "K-Means": [
        kmeans_count.get(0, 0),
        kmeans_count.get(1, 0),
        kmeans_count.get(2, 0)
    ],
    "Gaussian Mixture": [
        gmm_count.get(0, 0),
        gmm_count.get(1, 0),
        gmm_count.get(2, 0)
    ],
    "Hierarchical": [
        hierarchical_count.get(0, 0),
        hierarchical_count.get(1, 0),
        hierarchical_count.get(2, 0)
    ]
})

cluster_data = pd.DataFrame({
    "Hours Studied": df["Hours_Studied"],
    "Attendance": df["Attendance"],
    "Sleep Hours": df["Sleep_Hours"],
    "Previous Scores": df["Previous_Scores"],
    "K-Means Cluster": kmeans_labels + 1,
    "GMM Cluster": gmm_labels + 1,
    "Hierarchical Cluster": hierarchical_labels + 1
})

root = tk.Tk()
root.title("Experiment 5 - Clustering Algorithms")
root.geometry("1200x750")

style = ttk.Style()

style.configure(
    "Treeview",
    rowheight=28,
    font=("Arial", 10)
)

style.configure(
    "Treeview.Heading",
    font=("Arial", 10, "bold")
)

title = tk.Label(
    root,
    text="EXPERIMENT 5: CLUSTERING ALGORITHMS",
    font=("Arial", 18, "bold")
)

title.pack(pady=12)

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=15,
    pady=10
)


def create_table(parent, data):

    frame = ttk.Frame(parent)

    frame.pack(
        fill="both",
        expand=True
    )

    columns = list(data.columns)

    tree = ttk.Treeview(
        frame,
        columns=columns,
        show="headings"
    )

    vertical_scroll = ttk.Scrollbar(
        frame,
        orient="vertical",
        command=tree.yview
    )

    horizontal_scroll = ttk.Scrollbar(
        frame,
        orient="horizontal",
        command=tree.xview
    )

    tree.configure(
        yscrollcommand=vertical_scroll.set,
        xscrollcommand=horizontal_scroll.set
    )

    for column in columns:

        tree.heading(
            column,
            text=column
        )

        tree.column(
            column,
            width=170,
            minwidth=100,
            anchor="center"
        )

    for _, row in data.iterrows():

        tree.insert(
            "",
            "end",
            values=list(row)
        )

    tree.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    vertical_scroll.grid(
        row=0,
        column=1,
        sticky="ns"
    )

    horizontal_scroll.grid(
        row=1,
        column=0,
        sticky="ew"
    )

    frame.grid_rowconfigure(
        0,
        weight=1
    )

    frame.grid_columnconfigure(
        0,
        weight=1
    )


comparison_frame = ttk.Frame(notebook)

notebook.add(
    comparison_frame,
    text="Algorithm Comparison"
)

create_table(
    comparison_frame,
    result_data
)


count_frame = ttk.Frame(notebook)

notebook.add(
    count_frame,
    text="Cluster Counts"
)

create_table(
    count_frame,
    cluster_count_data
)


cluster_frame = ttk.Frame(notebook)

notebook.add(
    cluster_frame,
    text="Cluster Results"
)

create_table(
    cluster_frame,
    cluster_data
)


graph_frame = ttk.Notebook(notebook)

notebook.add(
    graph_frame,
    text="Visualizations"
)


kmeans_frame = ttk.Frame(graph_frame)

graph_frame.add(
    kmeans_frame,
    text="K-Means"
)

fig1, ax1 = plt.subplots(
    figsize=(7, 5)
)

ax1.scatter(
    df["Hours_Studied"],
    df["Attendance"],
    c=kmeans_labels,
    s=15
)

ax1.set_title(
    "K-Means Clustering",
    fontsize=14,
    fontweight="bold"
)

ax1.set_xlabel(
    "Hours Studied"
)

ax1.set_ylabel(
    "Attendance"
)

ax1.grid(True)

fig1.tight_layout()

canvas1 = FigureCanvasTkAgg(
    fig1,
    master=kmeans_frame
)

canvas1.draw()

canvas1.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


gmm_frame = ttk.Frame(graph_frame)

graph_frame.add(
    gmm_frame,
    text="Gaussian Mixture"
)

fig2, ax2 = plt.subplots(
    figsize=(7, 5)
)

ax2.scatter(
    df["Hours_Studied"],
    df["Attendance"],
    c=gmm_labels,
    s=15
)

ax2.set_title(
    "Gaussian Mixture Model",
    fontsize=14,
    fontweight="bold"
)

ax2.set_xlabel(
    "Hours Studied"
)

ax2.set_ylabel(
    "Attendance"
)

ax2.grid(True)

fig2.tight_layout()

canvas2 = FigureCanvasTkAgg(
    fig2,
    master=gmm_frame
)

canvas2.draw()

canvas2.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


hierarchical_frame = ttk.Frame(graph_frame)

graph_frame.add(
    hierarchical_frame,
    text="Hierarchical"
)

fig3, ax3 = plt.subplots(
    figsize=(7, 5)
)

ax3.scatter(
    df["Hours_Studied"],
    df["Attendance"],
    c=hierarchical_labels,
    s=15
)

ax3.set_title(
    "Hierarchical Clustering",
    fontsize=14,
    fontweight="bold"
)

ax3.set_xlabel(
    "Hours Studied"
)

ax3.set_ylabel(
    "Attendance"
)

ax3.grid(True)

fig3.tight_layout()

canvas3 = FigureCanvasTkAgg(
    fig3,
    master=hierarchical_frame
)

canvas3.draw()

canvas3.get_tk_widget().pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


root.mainloop()
