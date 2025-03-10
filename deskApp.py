import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import tkinterLib as tLib
import regressions as regLib

notebookFrameNames = ["Linear Regression", "Logistic Regression"]

def browseFiles():
    filename = filedialog.askopenfilename(initialdir=f"{os.getcwd}", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if filename:  # Proceed only if a file is selected
        df = pd.read_csv(filename)
        label_file_explorer.config(text="File Opened: " + filename)
        display_table(df)

        selected_tab = notebook.index(notebook.select())  # Get index of selected tab
    
        match selected_tab: 
            case 0:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                train_x, test_x, train_y, test_y, pred_y = regLib.prepAndTrainCSVForLinearRegression(df)

                # Calculate regression error metrics
                mae = mean_absolute_error(test_y, pred_y)
                mse = mean_squared_error(test_y, pred_y)
                rmse = mse ** 0.5  # Square root of MSE
                r2 = r2_score(test_y, pred_y)  # R² score (goodness of fit)

                tLib.BrowseFilesDisplayErros(linear_frame, tk, mae, mse, rmse, r2)
            case 1:
                print("hewwo")
            case _: 
                print("default/no match")
            


def display_table(df):
    # Clear existing table (if any)
    for widget in frame_table.winfo_children():
        widget.destroy()

    # Create Treeview inside frame_table
    tree = ttk.Treeview(frame_table)

    # Define Columns
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"  # Hide default empty first column

    # Configure column headers and make them expand dynamically
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")  # Center text in columns

    # Insert rows
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Pack Treeview with expansion
    tree.pack(expand=True, fill="both")

    # Function to adjust column widths dynamically
    def adjust_column_widths(event):
        total_width = frame_table.winfo_width()
        num_columns = len(df.columns)
        col_width = max(total_width // num_columns, 100)  # Minimum width = 100px
        for col in df.columns:
            tree.column(col, width=col_width)

    # Bind resizing event to adjust column widths
    frame_table.bind("<Configure>", adjust_column_widths)


# Tkinter Window Setup
window = tk.Tk()
window.title("Paw Pularity")
window.geometry("800x500")

# to make tabs (is a widget)
notebook = ttk.Notebook(window)
notebook.pack(padx=10, pady=10, expand=True, fill="both")

# linear tab
linear_frame = ttk.Frame(notebook)
linear_frame.pack(fill="both", expand=True)

# Configure frame_table to expand with window
linear_frame.columnconfigure(0, weight=1)  # Expand first column
linear_frame.rowconfigure(4, weight=1)     # Expand row containing frame_table

# logistic tab
logistic_frame = ttk.Frame(notebook)
linear_frame.pack(fill="both", expand=True)

# Frame for Table Display (expands with window)
frame_table = tk.Frame(linear_frame)
frame_table.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="nsew")


label_file_explorer = tLib.HeaderDisplay(linear_frame, tk, browseFiles, window)

# Add tabs to the Notebook widget
notebook.add(linear_frame, text=notebookFrameNames[0])
notebook.add(logistic_frame, text=notebookFrameNames[1])

window.mainloop()