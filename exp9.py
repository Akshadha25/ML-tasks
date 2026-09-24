import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

adaboost = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

logistic = LogisticRegression(
    max_iter=1000
)

svm = SVC(
    probability=True,
    random_state=42
)

voting = VotingClassifier(
    estimators=[
        ("logistic", logistic),
        ("svm", svm),
        ("random_forest", random_forest)
    ],
    voting="soft"
)

models = {
    "Random Forest": random_forest,
    "AdaBoost": adaboost,
    "Voting Classifier": voting
}

results = []

predictions = {}

for name, model in models.items():

    if name == "Voting Classifier":

        model.fit(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

    predictions[name] = prediction

    results.append([
        name,
        round(
            accuracy_score(
                y_test,
                prediction
            ),
            4
        ),
        round(
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        ),
        round(
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        ),
        round(
            f1_score(
                y_test,
                prediction,
                zero_division=0
            ),
            4
        )
    ])

result_data = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

prediction_data = pd.DataFrame({
    "Actual Result": [
        "High Score" if x == 1 else "Low Score"
        for x in y_test.values
    ],
    "Random Forest": [
        "High Score" if x == 1 else "Low Score"
        for x in predictions["Random Forest"]
    ],
    "AdaBoost": [
        "High Score" if x == 1 else "Low Score"
        for x in predictions["AdaBoost"]
    ],
    "Voting Classifier": [
        "High Score" if x == 1 else "Low Score"
        for x in predictions["Voting Classifier"]
    ]
})

root = tk.Tk()

root.title(
    "Experiment 9 - Ensemble Learning"
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
    text="EXPERIMENT 9: ENSEMBLE LEARNING",
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
            width=170,
            minwidth=110,
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
    result_data
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

graph_frame = ttk.Frame(notebook)

notebook.add(
    graph_frame,
    text="Accuracy Comparison"
)

fig, ax = plt.subplots(
    figsize=(7, 4)
)

ax.bar(
    result_data["Model"],
    result_data["Accuracy"]
)

ax.set_title(
    "Ensemble Model Accuracy"
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_ylim(
    0,
    1
)

ax.tick_params(
    axis="x",
    rotation=15
)

ax.grid(
    axis="y"
)

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