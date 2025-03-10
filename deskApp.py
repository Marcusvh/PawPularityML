import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

        # adding labels to show the result.
        title_label = tk.Label(window, text="Model Performance Metrics", font=("Arial", 14, "bold"))
        title_label.grid(column=0, row=5, columnspan=5, pady=10)

        LinearText = tk.Label(window, text="Errors when using LinearRegression", font=("Arial", 12, "italic"))
        LinearText.grid(column=0, row=6, columnspan=5, pady=5)

        r2Score = tk.Label(window, text=f"R² Score: {r2:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
        r2Score.grid(column=0, row=7, sticky="w", pady=5)

        rmseScore = tk.Label(window, text=f"Root Mean Squared Error (RMSE): {rmse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
        rmseScore.grid(column=0, row=8, sticky="w", pady=5)

        mseScore = tk.Label(window, text=f"Mean Squared Error (MSE): {mse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
        mseScore.grid(column=0, row=9, sticky="w", pady=5)

        maeScore = tk.Label(window, text=f"Mean Absolute Error (MAE): {mae:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
        maeScore.grid(column=0, row=10, sticky="w", pady=5)

        # extra config
        r2Score.config(bg="lightblue")
        rmseScore.config(bg="lightyellow")
        mseScore.config(bg="lightgreen")
        maeScore.config(bg="lightcoral")

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

label_file_explorer = tk.Label(window, text="File Explorer using Tkinter", width=100, height=2, fg="blue")
label_file_explorer.grid(column=1, row=1)

button_explore = tk.Button(window, text="Browse Files", command=browseFiles)
button_explore.grid(column=1, row=2)

button_exit = tk.Button(window, text="Exit", command=window.quit)
button_exit.grid(column=1, row=3)

# Frame for Table Display
frame_table = tk.Frame(window)
frame_table.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="nsew")

window.mainloop()