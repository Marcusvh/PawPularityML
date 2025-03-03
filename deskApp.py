import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os

def browseFiles():
    filename = filedialog.askopenfilename(initialdir=f"{os.getcwd}", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))

    if filename:  # Proceed only if a file is selected
        df = pd.read_csv(filename)
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

newLabel = tk.Label(window, text="Testo")
newLabel.grid(column=0, row=0)

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
