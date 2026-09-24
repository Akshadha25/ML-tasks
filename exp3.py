import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

X = df[["Hours_Studied"]]
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

summary_data = pd.DataFrame({
    "Property": [
        "Input Feature",
        "Target",
        "Training Samples",
        "Testing Samples",
        "Slope",
        "Intercept",
        "Mean Squared Error",
        "R² Score"
    ],
    "Value": [
        "Hours_Studied",
        "Exam_Score",
        len(X_train),
        len(X_test),
        round(model.coef_[0], 2),
        round(model.intercept_, 2),
        round(mse, 2),
        round(r2, 2)
    ]
})

prediction_data = pd.DataFrame({
    "Hours Studied": X_test["Hours_Studied"].values,
    "Actual Exam Score": y_test.values,
    "Predicted Exam Score": y_pred.round(2)
})

root = tk.Tk()
root.title("Experiment 3 - Linear Regression")
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
    text="EXPERIMENT 3: LINEAR REGRESSION",
    font=("Arial", 16, "bold")
)

title.pack(pady=10)

notebook = ttk.Notebook(root)

notebook.pack(
    fill="both",
    expand=True,
    padx=12,
    pady=8
)

def create_table(parent, data):
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)

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

summary_frame = ttk.Frame(notebook)

notebook.add(
    summary_frame,
    text="Model Summary"
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
    prediction_data
)

graph_frame = ttk.Frame(notebook)

notebook.add(
    graph_frame,
    text="Regression Graph"
)

fig, ax = plt.subplots(
    figsize=(6, 3.5)
)

ax.scatter(
    X_test["Hours_Studied"],
    y_test,
    label="Actual Values",
    s=25
)

sorted_indices = X_test["Hours_Studied"].values.argsort()

ax.plot(
    X_test["Hours_Studied"].values[sorted_indices],
    y_pred[sorted_indices],
    color="red",
    linewidth=2,
    label="Regression Line"
)

ax.set_title(
    "Hours Studied vs Exam Score",
    fontsize=11
)

ax.set_xlabel(
    "Hours Studied",
    fontsize=9
)

ax.set_ylabel(
    "Exam Score",
    fontsize=9
)

ax.tick_params(
    axis="both",
    labelsize=8
)

ax.legend(
    fontsize=8
)

ax.grid(True)

fig.tight_layout(
    pad=1
)

canvas = FigureCanvasTkAgg(
    fig,
    master=graph_frame
)

canvas.draw()

canvas.get_tk_widget().pack(
    fill="none",
    expand=False,
    padx=20,
    pady=15
)

root.mainloop()