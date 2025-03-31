import tkinter as tk
from tkinter import ttk
import tkinterLib as tLib
import models as regLib
import tkinterWindowSetup as tWin
from sklearn.datasets import load_breast_cancer

notebookFrameNames = ["Linear Regression", "Logistic Regression"]
train_csv_path = "./train.csv"

def browseFiles():
    # Load the breast cancer dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    label_file_explorer.config(text="Breast Cancer Dataset Loaded")
    
    selected_tab = notebook.index(notebook.select())  # Get index of selected tab

    match selected_tab: 
        case 0:
            train_x, test_x, train_y, test_y = regLib.prepRegressionData(df)
            pred_y, model = regLib.PredictLinearRegression(train_x, train_y, test_x)

            mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=test_y, pred_y=pred_y)

            display_table(test_x.assign(Predicted=pred_y), notebookFrameNames[0])
            tLib.displayRegressionErrors(linear_frame, tk, mae, mse, rmse, r2)

            cv_scores = regLib.crossValidationScores(model, train_x, train_y)
            tLib.displayCrossValidationScores(linear_frame, tk, cv_scores)

        case 1:
            model, pred_y, accuracy = regLib.PredictLogisticRegression(df)  # Adjusted unpacking

            mae, mse, rmse, r2 = regLib.regressionErrorScores(test_y=df["target"], pred_y=pred_y)
            tLib.displayRegressionErrors(logistic_frame, tk, mae, mse, rmse, r2)
            tLib.displayAccuracyScore(logistic_frame, tk, accuracy)

            display_table(df.assign(Predicted=pred_y), notebookFrameNames[1])

            cv_scores = regLib.crossValidationScores(model, df.drop(columns=["target"]), df["target"])
            tLib.displayCrossValidationScores(logistic_frame, tk, cv_scores)

            fpr, tpr, roc_auc = regLib.calculateROCCurve(model, df.drop(columns=["target"]), df["target"])
            tLib.displayROCCurve(logistic_frame, fpr, tpr, roc_auc)

        case _: 
            print("default/no match")

def display_table(df, tab_name):
    """Creates a Treeview table with vertical and horizontal scrolling inside a scrollable frame."""
    if tab_name not in frame_tables:
        print(f"Error: No frame_table found for {tab_name}")
        return

    frame_table = frame_tables[tab_name]

    # Clear previous content
    for widget in frame_table.winfo_children():
        widget.destroy()

    # Create a Canvas for scrolling
    canvas = tk.Canvas(frame_table)
    scrollable_frame = ttk.Frame(canvas)

    # Scrollbars
    scrollbar_y = ttk.Scrollbar(frame_table, orient="vertical", command=canvas.yview)
    scrollbar_x = ttk.Scrollbar(frame_table, orient="horizontal", command=canvas.xview)

    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    # Pack everything
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    canvas.pack(side="left", fill="both", expand=True)

    # Create a window inside the canvas
    window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Table inside scrollable frame
    tree = ttk.Treeview(scrollable_frame, columns=list(df.columns), show="headings")

    # Define column headers
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", minwidth=100, width=150)  

    # Insert rows into table
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(fill="both", expand=True)

    # Update scroll region
    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", update_scroll_region)

    # Enable mousewheel scrolling
    def on_mouse_wheel(event):
        canvas.xview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind_all("<MouseWheel>", on_mouse_wheel)  # Windows/macOS
    canvas.bind_all("<Shift-MouseWheel>", on_mouse_wheel)  # Horizontal scrolling


window = tWin.setup_window()
notebook = tWin.setup_notebook(window, ttk)

linear_container, linear_frame, logistic_container, logistic_frame = tWin.setup_tabs(notebook, ttk)
frame_tables, label_file_explorer = tWin.setup_frames(linear_frame, logistic_frame, tLib, browseFiles, window)    
tWin.add_tabs_to_notebook(notebook, linear_container, logistic_container, notebookFrameNames)


window.mainloop()