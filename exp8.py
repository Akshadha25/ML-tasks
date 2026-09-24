import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv(
    "/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv"
)

median_score = df["Exam_Score"].median()

df["Result"] = (
    df["Exam_Score"] >= median_score
).astype(int)

features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores"
]

X = df[features]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

summary_data = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4)
    ]
})

prediction_data = pd.DataFrame({
    "Actual Result": [
        "High Score" if x == 1 else "Low Score"
        for x in y_test.values
    ],
    "Predicted Result": [
        "High Score" if x == 1 else "Low Score"
        for x in y_pred
    ]
})

root = tk.Tk()

root.title(
    "Experiment 8 - CART"
)

root.geometry(
    "1100x700"
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
    text="EXPERIMENT 8: CART CLASSIFICATION",
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

    scrollbar = ttk.Scrollbar(
        frame,
        orient="vertical",
        command=tree.yview
    )

    tree.configure(
        yscrollcommand=scrollbar.set
    )

    for column in columns:

        tree.heading(
            column,
            text=column
        )

        tree.column(
            column,
            width=220,
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

    scrollbar.grid(
        row=0,
        column=1,
        sticky="ns"
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
    text="Model Performance"
)

create_table(
    summary_frame,
    summary_data
)

prediction_frame = ttk.Frame(notebook)

notebook.add(
    prediction_frame,
    text="Predictions"
)

create_table(
    prediction_frame,
    prediction_data.head(100)
)

tree_frame = ttk.Frame(notebook)

notebook.add(
    tree_frame,
    text="Decision Tree"
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

plot_tree(
    model,
    feature_names=features,
    class_names=[
        "Low Score",
        "High Score"
    ],
    filled=True,
    ax=ax
)

fig.tight_layout()

canvas = FigureCanvasTkAgg(
    fig,
    master=tree_frame
)

canvas.draw()

canvas.get_tk_widget().pack(
    padx=10,
    pady=10
)

cm_frame = ttk.Frame(notebook)

notebook.add(
    cm_frame,
    text="Confusion Matrix"
)

fig2, ax2 = plt.subplots(
    figsize=(5, 4)
)

ax2.imshow(
    cm,
    cmap="Blues"
)

ax2.set_title(
    "CART Confusion Matrix"
)

ax2.set_xlabel(
    "Predicted"
)

ax2.set_ylabel(
    "Actual"
)

ax2.set_xticks(
    [0, 1]
)

ax2.set_yticks(
    [0, 1]
)

for i in range(2):

    for j in range(2):

        ax2.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

fig2.tight_layout()

canvas2 = FigureCanvasTkAgg(
    fig2,
    master=cm_frame
)

canvas2.draw()

canvas2.get_tk_widget().pack(
    padx=20,
    pady=20
)

root.mainloop()