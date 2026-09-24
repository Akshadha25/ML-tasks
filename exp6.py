import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

explained_variance = pca.explained_variance_ratio_

variance_data = pd.DataFrame({
    "Principal Component": ["PC1", "PC2"],
    "Explained Variance": [
        explained_variance[0],
        explained_variance[1]
    ],
    "Percentage": [
        explained_variance[0] * 100,
        explained_variance[1] * 100
    ]
})

root = tk.Tk()

root.title("Experiment 6 - PCA")
root.geometry("1100x700")

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
    text="EXPERIMENT 6: PRINCIPAL COMPONENT ANALYSIS",
    font=("Arial", 16, "bold")
)

title.pack(pady=10)

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
            width=200,
            anchor="center"
        )

    for _, row in data.iterrows():

        tree.insert(
            "",
            "end",
            values=[
                round(value, 4) if isinstance(value, float) else value
                for value in row
            ]
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

summary_frame = ttk.Frame(notebook)

notebook.add(
    summary_frame,
    text="PCA Summary"
)

create_table(
    summary_frame,
    variance_data
)

data_frame = ttk.Frame(notebook)

notebook.add(
    data_frame,
    text="Transformed Data"
)

create_table(
    data_frame,
    pca_data.head(100)
)

graph_frame = ttk.Frame(notebook)

notebook.add(
    graph_frame,
    text="PCA Graph"
)

fig, ax = plt.subplots(
    figsize=(5.5, 3.8)
)

ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=15
)

ax.set_title(
    "PCA - Two Principal Components",
    fontsize=12,
    fontweight="bold"
)

ax.set_xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}%)"
)

ax.set_ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}%)"
)

ax.grid(True)

fig.tight_layout()

canvas = FigureCanvasTkAgg(
    fig,
    master=graph_frame
)

canvas.draw()

canvas.get_tk_widget().pack(
    padx=20,
    pady=15
)

root.mainloop()