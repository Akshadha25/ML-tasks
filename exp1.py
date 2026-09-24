import pandas as pd
import tkinter as tk
from tkinter import ttk

df = pd.read_csv("/Users/akshadha/Desktop/ML_lab/StudentPerformanceFactors.csv")

root = tk.Tk()
root.title("Experiment 1 - Dataset Exploration")
root.geometry("1200x750")

style = ttk.Style()
style.configure("Treeview", rowheight=30, font=("Arial", 11))
style.configure("Treeview.Heading", font=("Arial", 11, "bold"))

title = tk.Label(
    root,
    text="EXPERIMENT 1: DATASET EXPLORATION",
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

first_frame = ttk.Frame(notebook)
notebook.add(first_frame, text="First 5 Rows")

create_table(first_frame, df.head())

last_frame = ttk.Frame(notebook)
notebook.add(last_frame, text="Last 5 Rows")

create_table(last_frame, df.tail())

dimension_frame = ttk.Frame(notebook)
notebook.add(dimension_frame, text="Dimensions")

dimension_data = pd.DataFrame({
    "Property": [
        "Number of Rows",
        "Number of Columns"
    ],
    "Value": [
        df.shape[0],
        df.shape[1]
    ]
})

create_table(dimension_frame, dimension_data)

columns_frame = ttk.Frame(notebook)
notebook.add(columns_frame, text="Column Names")

column_data = pd.DataFrame({
    "S. No.": range(1, len(df.columns) + 1),
    "Column Name": df.columns
})

create_table(columns_frame, column_data)

dtype_frame = ttk.Frame(notebook)
notebook.add(dtype_frame, text="Data Types")

dtype_data = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": [
        str(df[column].dtype)
        for column in df.columns
    ]
})

create_table(dtype_frame, dtype_data)

root.mainloop()