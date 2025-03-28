import tkinter as tk

def setup_window():
    window = tk.Tk()
    window.title("Paw Pularity")
    window.geometry("800x500")
    return window

def setup_notebook(window, ttk):
    notebook = ttk.Notebook(window)
    notebook.pack(padx=10, pady=10, expand=True, fill="both")
    return notebook

def setup_tabs(notebook, ttk):
    linear_container, linear_frame = create_scrollable_frame(notebook, ttk)
    linear_frame.pack(fill="both", expand=True)
    linear_frame.columnconfigure(0, weight=1)
    linear_frame.rowconfigure(4, weight=1)

    logistic_container, logistic_frame = create_scrollable_frame(notebook, ttk)
    logistic_frame.columnconfigure(0, weight=1)
    logistic_frame.rowconfigure(4, weight=1)
    logistic_frame.pack(fill="both", expand=True)

    return linear_container, linear_frame, logistic_container, logistic_frame

def setup_frames(linear_frame, logistic_frame, tLib, browseFiles, window):
    frames_with_file_explorer = [linear_frame, logistic_frame]
    frame_tables = {
        "Linear Regression": tk.Frame(linear_frame),
        "Logistic Regression": tk.Frame(logistic_frame)
    }

    for name, frame_table in frame_tables.items():
        frame_table.grid(column=0, row=4, columnspan=3, padx=10, pady=10, sticky="nsew")

    for frame in frames_with_file_explorer:
        label_file_explorer = tLib.headerDisplay(frame, tk, browseFiles, window)

    return frame_tables, label_file_explorer

def add_tabs_to_notebook(notebook, linear_container, logistic_container, notebookFrameNames):
    notebook.add(linear_container, text=notebookFrameNames[0])
    notebook.add(logistic_container, text=notebookFrameNames[1])


def create_scrollable_frame(parent, ttk):
    container = ttk.Frame(parent)
    canvas = tk.Canvas(container)
    scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollbar_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)

    # Frame that will contain your content
    scroll_frame = ttk.Frame(canvas)

    # Update scroll region when content changes
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    # Layout management
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar_y.grid(row=0, column=1, sticky="ns")
    scrollbar_x.grid(row=1, column=0, sticky="ew")

    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    return container, scroll_frame
