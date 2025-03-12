import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os

import tkinterLib as tLib
import models as regLib

notebookFrameNames = ["Linear Regression", "Logistic Regression"]

def browseFiles():
    filename = filedialog.askopenfilename(initialdir=f"{os.getcwd}", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if filename:  # Proceed only if a file is selected
        df = pd.read_csv(filename)
        label_file_explorer.config(text="File Opened: " + filename)
        
        selected_tab = notebook.index(notebook.select())  # Get index of selected tab
    
        match selected_tab: 
            case 0:
                test_y, pred_y = regLib.prepAndTrainCSVForLinearRegression(df)

                mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=test_y, pred_y=pred_y)

                display_table(df, notebookFrameNames[0])
                tLib.displayRegressionErrors(linear_frame, tk, mae, mse, rmse, r2)

            case 1:
                test_y, pred_y = regLib.prepAndTrainCSVForLogisticRegression(df)
                mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=test_y, pred_y=pred_y)
                tLib.displayRegressionErrors(logistic_frame, tk, mae, mse, rmse, r2)
                
                display_table(df, notebookFrameNames[1])
                print("hewwo")
            case _: 
                print("default/no match")
            


def display_table(df, tab_name):
    """Displays a Pandas DataFrame in the correct tab."""
    if tab_name not in frame_tables:
        print(f"Error: No frame_table found for {tab_name}")
        return

    frame_table = frame_tables[tab_name]

    # Clear previous table
    for widget in frame_table.winfo_children():
        widget.destroy()

    tree = ttk.Treeview(frame_table)
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"

    # Add column headers
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    # Insert data rows
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill="both")

    # Adjust column widths dynamically
    def adjust_column_widths(event):
        total_width = frame_table.winfo_width()
        num_columns = len(df.columns)
        col_width = max(total_width // num_columns, 100)
        for col in df.columns:
            tree.column(col, width=col_width)

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
linear_frame.columnconfigure(0, weight=1)
linear_frame.rowconfigure(4, weight=1)

# logistic tab
logistic_frame = ttk.Frame(notebook)
logistic_frame.columnconfigure(0, weight=1)
logistic_frame.rowconfigure(4, weight=1)
logistic_frame.pack(fill="both", expand=True)

frames_with_file_explorer = [linear_frame, logistic_frame]
# Store frames using names as keys instead of direct frame objects
frame_tables = {
    "Linear Regression": tk.Frame(linear_frame),
    "Logistic Regression": tk.Frame(logistic_frame)
}

# Loop through the dictionary to place them correctly
for name, frame_table in frame_tables.items():
    frame_table.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="nsew")

for frame in frames_with_file_explorer:
    label_file_explorer = tLib.headerDisplay(frame, tk, browseFiles, window)
    

# Add tabs to the Notebook widget
notebook.add(linear_frame, text=notebookFrameNames[0])
notebook.add(logistic_frame, text=notebookFrameNames[1])

window.mainloop()