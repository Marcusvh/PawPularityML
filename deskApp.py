import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os
import tkinterLib as tLib
import models as regLib
import tkinterWindowSetup as tWin

notebookFrameNames = ["Linear Regression", "Logistic Regression"]
train_csv_path = "./train.csv"

def browseFiles():
    filename = filedialog.askopenfilename(initialdir=f"{os.getcwd}", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if filename:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(filename)
        label_file_explorer.config(text="File Opened: " + filename)
        
        selected_tab = notebook.index(notebook.select())  # Get index of selected tab
    
        match selected_tab: 
            case 0:
                train_x, test_x, train_y, test_y = regLib.prepRegressionData(train_df)
                pred_y, model = regLib.trainAndPredictLinearRegression(train_x, train_y, test_x)
                regLib.predictModelRegression(model, test_df) # prediction for user selected csv file, based on model trained on train.csv

                mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=test_y, pred_y=pred_y)

                display_table(test_df, notebookFrameNames[0])
                tLib.displayRegressionErrors(linear_frame, tk, mae, mse, rmse, r2)

                cv_scores = regLib.crossValidationScores(model, train_x, train_y)
                tLib.displayCrossValidationScores(linear_frame, tk, cv_scores)

            case 1:
                train_x, test_x, train_y, test_y = regLib.prepRegressionData(train_df)
                pred_y, model = regLib.trainAndPredictLogisticRegression(train_x, train_y, test_x)
                regLib.predictModelRegression(model, test_df) # prediction for user selected csv file, based on model trained on train.csv

                mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=test_y, pred_y=pred_y)
                tLib.displayRegressionErrors(logistic_frame, tk, mae, mse, rmse, r2)
                
                display_table(test_df, notebookFrameNames[1])

                cv_scores = regLib.crossValidationScores(model, train_x, train_y)
                tLib.displayCrossValidationScores(logistic_frame, tk, cv_scores)

                fpr, tpr, roc_auc = regLib.calculateROCCurve(model, test_x, test_y)
                tLib.displayROCCurve(logistic_frame, tk, fpr, tpr, roc_auc)
                
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

window = tWin.setup_window()
notebook = tWin.setup_notebook(window, ttk)

linear_container, linear_frame, logistic_container, logistic_frame = tWin.setup_tabs(notebook, ttk)
frame_tables, label_file_explorer = tWin.setup_frames(linear_frame, logistic_frame, tLib, browseFiles, window)    
tWin.add_tabs_to_notebook(notebook, linear_container, logistic_container, notebookFrameNames)


window.mainloop()