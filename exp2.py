import pandas as pd
import tkinter as tk
from tkinter import ttk

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

root = tk.Tk()
root.title("Experiment 2 - Summary and Statistics")
root.geometry("1200x750")

style = ttk.Style()
style.configure("Treeview", rowheight=30, font=("Arial", 11))
style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

title = tk.Label(
    root,
    text="EXPERIMENT 2: SUMMARY AND STATISTICS",
    font=("Arial", 18, "bold")
)
title.pack(pady=15)

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=15, pady=10)


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
        tree.heading(column, text=column)
        tree.column(column, width=150, minwidth=100, anchor="center")

    for _, row in data.iterrows():
        tree.insert("", "end", values=list(row))

    tree.grid(row=0, column=0, sticky="nsew")
    vertical_scroll.grid(row=0, column=1, sticky="ns")
    horizontal_scroll.grid(row=1, column=0, sticky="ew")

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    return tree


numeric_df = df.select_dtypes(include="number")


overview_frame = ttk.Frame(notebook)
notebook.add(overview_frame, text="Dataset Overview")

overview_data = pd.DataFrame({
    "Property": [
        "Number of Rows",
        "Number of Columns",
        "Numerical Columns",
        "Categorical Columns",
        "Missing Values"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        len(df.select_dtypes(include="number").columns),
        len(df.select_dtypes(exclude="number").columns),
        df.isnull().sum().sum()
    ]
})

create_table(overview_frame, overview_data)


summary_frame = ttk.Frame(notebook)
notebook.add(summary_frame, text="Statistical Summary")

summary_data = numeric_df.describe().round(2).T.reset_index()
summary_data.rename(columns={"index": "Column"}, inplace=True)

create_table(summary_frame, summary_data)


mean_frame = ttk.Frame(notebook)
notebook.add(mean_frame, text="Mean")

mean_data = pd.DataFrame({
    "Column": numeric_df.columns,
    "Mean": numeric_df.mean().round(2)
})

create_table(mean_frame, mean_data)


median_frame = ttk.Frame(notebook)
notebook.add(median_frame, text="Median")

median_data = pd.DataFrame({
    "Column": numeric_df.columns,
    "Median": numeric_df.median().round(2)
})

create_table(median_frame, median_data)


std_frame = ttk.Frame(notebook)
notebook.add(std_frame, text="Standard Deviation")

std_data = pd.DataFrame({
    "Column": numeric_df.columns,
    "Standard Deviation": numeric_df.std().round(2)
})

create_table(std_frame, std_data)


missing_frame = ttk.Frame(notebook)
notebook.add(missing_frame, text="Missing Values")

missing_data = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values
})

create_table(missing_frame, missing_data)


root.mainloop()