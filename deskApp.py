import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tkinterLib as tLib

def browseFiles():
    filename = filedialog.askopenfilename(initialdir=f"{os.getcwd}", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if filename:  # Proceed only if a file is selected
        df = pd.read_csv(filename)

        # Prepare X and y
        X = df.iloc[:,1:-1]
        y = df.iloc[:,-1]

        # Do train test split
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = LinearRegression()
        reg.fit(train_x, train_y)

        pred_y = reg.predict(test_x)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Calculate regression metrics
        mae = mean_absolute_error(test_y, pred_y)
        mse = mean_squared_error(test_y, pred_y)
        rmse = mse ** 0.5  # Square root of MSE
        r2 = r2_score(test_y, pred_y)  # R² score (goodness of fit)

        tLib.BrowseFilesDisplayErros(linear_frame, tk, mae, mse, rmse, r2)
        
        label_file_explorer.config(text="File Opened: " + filename)
        display_table(df)

def display_table(df):
    # Clear existing table (if any)
    for widget in frame_table.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(frame_table)

    # Define Columns
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"  # Hide default empty first column

    # Add column headers
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Set column width (adjust as needed)

    # Insert rows
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill="both")



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

# logistic tab
logistic_frame = ttk.Frame(notebook)
linear_frame.pack(fill="both", expand=True)

# File Explorer label
label_file_explorer = tk.Label(linear_frame, text="File Explorer using Tkinter", width=100, height=2, fg="blue")
label_file_explorer.grid(column=0, row=1, columnspan=5, pady=10)

button_explore = tk.Button(linear_frame, text="Browse Files", command=browseFiles)
button_explore.grid(column=0, row=2, columnspan=5, pady=10)

button_exit = tk.Button(linear_frame, text="Exit", command=window.quit)
button_exit.grid(column=0, row=3, columnspan=5, pady=10)

# Frame for Table Display
frame_table = tk.Frame(linear_frame)
frame_table.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="nsew")

# Add tabs to the Notebook widget
notebook.add(linear_frame, text="Linear Regression")
notebook.add(logistic_frame, text="Logistic Regression")

window.mainloop()