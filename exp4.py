import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_b = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_b = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]

weights = np.zeros(
    X_train_b.shape[1]
)

learning_rate = 0.01
prior_variance = 10.0
iterations = 3000

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

for i in range(iterations):

    probabilities = sigmoid(
        np.dot(
            X_train_b,
            weights
        )
    )

    error = (
        probabilities -
        y_train.values
    )

    gradient = (
        np.dot(
            X_train_b.T,
            error
        ) / len(y_train)
    )

    prior_gradient = (
        weights / prior_variance
    )

    prior_gradient[0] = 0

    total_gradient = (
        gradient +
        prior_gradient
    )

    weights -= (
        learning_rate *
        total_gradient
    )

bayesian_probabilities = sigmoid(
    np.dot(
        X_test_b,
        weights
    )
)

bayesian_pred = (
    bayesian_probabilities >= 0.5
).astype(int)

svm_model = SVC(
    kernel="rbf"
)

svm_model.fit(
    X_train_scaled,
    y_train
)

svm_pred = svm_model.predict(
    X_test_scaled
)

bayesian_accuracy = accuracy_score(
    y_test,
    bayesian_pred
)

bayesian_precision = precision_score(
    y_test,
    bayesian_pred,
    zero_division=0
)

bayesian_recall = recall_score(
    y_test,
    bayesian_pred,
    zero_division=0
)

bayesian_f1 = f1_score(
    y_test,
    bayesian_pred,
    zero_division=0
)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

svm_precision = precision_score(
    y_test,
    svm_pred,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_pred,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_pred,
    zero_division=0
)

comparison_data = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Bayesian Logistic Regression": [
        round(bayesian_accuracy, 4),
        round(bayesian_precision, 4),
        round(bayesian_recall, 4),
        round(bayesian_f1, 4)
    ],
    "SVM": [
        round(svm_accuracy, 4),
        round(svm_precision, 4),
        round(svm_recall, 4),
        round(svm_f1, 4)
    ]
})

prediction_data = pd.DataFrame({
    "Actual Result": [
        "High Score" if value == 1 else "Low Score"
        for value in y_test.values
    ],
    "Bayesian Logistic Regression": [
        "High Score" if value == 1 else "Low Score"
        for value in bayesian_pred
    ],
    "SVM": [
        "High Score" if value == 1 else "Low Score"
        for value in svm_pred
    ]
})

root = tk.Tk()

root.title(
    "Experiment 4 - Bayesian Logistic Regression and SVM"
)

root.geometry("1000x650")

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
    text="EXPERIMENT 4: BAYESIAN LOGISTIC REGRESSION AND SVM",
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
            width=190,
            minwidth=120,
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
    text="Model Comparison"
)

create_table(
    comparison_frame,
    comparison_data
)

prediction_frame = ttk.Frame(notebook)

notebook.add(
    prediction_frame,
    text="Predictions"
)

create_table(
    prediction_frame,
    prediction_data
)

confusion_frame = ttk.Frame(notebook)

notebook.add(
    confusion_frame,
    text="Confusion Matrices"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(6, 3.2)
)

bayesian_cm = confusion_matrix(
    y_test,
    bayesian_pred
)

svm_cm = confusion_matrix(
    y_test,
    svm_pred
)

axes[0].imshow(
    bayesian_cm,
    cmap="Reds"
)

axes[0].set_title(
    "Bayesian Logistic Regression",
    fontsize=10
)

axes[0].set_xlabel(
    "Predicted",
    fontsize=8
)

axes[0].set_ylabel(
    "Actual",
    fontsize=8
)

axes[0].set_xticks(
    [0, 1]
)

axes[0].set_yticks(
    [0, 1]
)

for i in range(2):

    for j in range(2):

        axes[0].text(
            j,
            i,
            bayesian_cm[i, j],
            ha="center",
            va="center",
            fontsize=9
        )

axes[1].imshow(
    svm_cm,
    cmap="Reds"
)

axes[1].set_title(
    "SVM",
    fontsize=10
)

axes[1].set_xlabel(
    "Predicted",
    fontsize=8
)

axes[1].set_ylabel(
    "Actual",
    fontsize=8
)

axes[1].set_xticks(
    [0, 1]
)

axes[1].set_yticks(
    [0, 1]
)

for i in range(2):

    for j in range(2):

        axes[1].text(
            j,
            i,
            svm_cm[i, j],
            ha="center",
            va="center",
            fontsize=9
        )

fig.tight_layout(
    pad=1
)

canvas = FigureCanvasTkAgg(
    fig,
    master=confusion_frame
)

canvas.draw()

canvas.get_tk_widget().pack(
    fill="none",
    expand=False,
    padx=20,
    pady=15
)

accuracy_frame = ttk.Frame(notebook)

notebook.add(
    accuracy_frame,
    text="Accuracy Comparison"
)

fig2, ax2 = plt.subplots(
    figsize=(6, 3.2)
)

models = [
    "Bayesian Logistic\nRegression",
    "SVM"
]

accuracies = [
    bayesian_accuracy,
    svm_accuracy
]

bars = ax2.bar(
    models,
    accuracies,
    width=0.5
)

ax2.set_title(
    "Model Accuracy Comparison",
    fontsize=11
)

ax2.set_ylabel(
    "Accuracy",
    fontsize=9
)

ax2.set_ylim(
    0,
    1
)

ax2.tick_params(
    axis="both",
    labelsize=8
)

for bar, value in zip(
    bars,
    accuracies
):

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )

ax2.grid(
    axis="y"
)

fig2.tight_layout(
    pad=1
)

canvas2 = FigureCanvasTkAgg(
    fig2,
    master=accuracy_frame
)

canvas2.draw()

canvas2.get_tk_widget().pack(
    fill="none",
    expand=False,
    padx=20,
    pady=15
)

root.mainloop()