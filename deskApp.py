import tkinter as tk
from tkinter import filedialog

def btnTesto():
    print("this is kinda confusing.")

def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/", 
                                          title="Select a file", 
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: "+filename)

window=tk.Tk()
window.title(" Paw Pularity ")
window.geometry("600x400")

newLabel = tk.Label(text= "Testo")
newLabel.grid(column=0, row=0)

button_name = tk.Button(window, text = "red")

button_name.bind("<Button-1>", btnTesto())


label_file_explorer = tk.Label(window, 
                            text = "File Explorer using Tkinter",
                            width = 100, height = 4, 
                            fg = "blue")
label_file_explorer.grid(column = 1, row = 1)

button_explore = tk.Button(window, 
                        text = "Browse Files",
                        command = browseFiles) 
button_explore.grid(column = 1, row = 2)
  
button_exit = tk.Button(window, 
                     text = "Exit",
                     command = exit) 
button_exit.grid(column = 1,row = 3)  

window.mainloop()