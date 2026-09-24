import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv(
    "/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv"
)

features = [
    "Hours_Studied",
    "Attendance"
]

X = df[features].head(500)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    random_state=42
)

model.fit(X_scaled)

hidden_states = model.predict(X_scaled)

result_data = pd.DataFrame({
    "Sequence": range(1, len(hidden_states) + 1),
    "Hours Studied": X["Hours_Studied"].values,
    "Attendance": X["Attendance"].values,
    "Hidden State": hidden_states
})

state_data = pd.DataFrame({
    "State": [
        0,
        1,
        2
    ],
    "State Count": [
        sum(hidden_states == 0),
        sum(hidden_states == 1),
        sum(hidden_states == 2)
    ]
})

root = tk.Tk()

root.title(
    "Experiment 7 - Hidden Markov Model"
)

root.geometry(
    "1000x650"
)

style = ttk.Style()

style.configure(
    "Treeview",
    rowheight=25,
    font=("Arial", 10)
)

style.configure(
    "Treeview.Heading",
    font=("Arial", 10, "bold")
)

title = tk.Label(
    root,
    text="EXPERIMENT 7: HIDDEN MARKOV MODEL",
    font=("Arial", 16, "bold")
)

title.pack(
    pady=10
)

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=8
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
            width=180,
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


prediction_frame = ttk.Frame(notebook)

notebook.add(
    prediction_frame,
    text="Hidden States"
)

create_table(
    prediction_frame,
    result_data.head(100)
)

state_frame = ttk.Frame(notebook)

notebook.add(
    state_frame,
    text="State Summary"
)

create_table(
    state_frame,
    state_data
)

graph_frame = ttk.Frame(notebook)

notebook.add(
    graph_frame,
    text="Sequence Graph"
)

fig, ax = plt.subplots(
    figsize=(7, 3.5)
)

ax.plot(
    range(1, len(hidden_states) + 1),
    hidden_states
)

ax.set_title(
    "HMM Hidden State Sequence"
)

ax.set_xlabel(
    "Sequence"
)

ax.set_ylabel(
    "Hidden State"
)

ax.set_yticks(
    [0, 1, 2]
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
    pady=20
)

root.mainloop()