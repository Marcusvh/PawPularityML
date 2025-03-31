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
    """Creates fully scrollable tabs using the smart frame approach."""
    linear_container, linear_frame = create_scrollable_frame(notebook, ttk)
    logistic_container, logistic_frame = create_scrollable_frame(notebook, ttk)

    notebook.add(linear_container, text="Linear Regression")
    notebook.add(logistic_container, text="Logistic Regression")

    return linear_container, linear_frame, logistic_container, logistic_frame



# def setup_tabs(notebook, ttk):
#     linear_container, linear_frame = create_scrollable_frame(notebook, ttk)
#     linear_frame.pack(fill="both", expand=True)
#     linear_frame.columnconfigure(0, weight=1)
#     linear_frame.rowconfigure(4, weight=1)

#     logistic_container, logistic_frame = create_scrollable_frame(notebook, ttk)
#     logistic_frame.columnconfigure(0, weight=1)
#     logistic_frame.rowconfigure(4, weight=1)
#     logistic_frame.pack(fill="both", expand=True)

#     return linear_container, linear_frame, logistic_container, logistic_frame

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
    """Creates a smart scrollable frame that allows natural scrolling for all elements inside."""
    container = ttk.Frame(parent)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

    # Create an inner frame for all content
    scroll_frame = ttk.Frame(canvas)

    # Update scroll region when frame changes size
    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Embed the frame inside the canvas
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set)

    # Layout packing
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar_y.pack(side="right", fill="y")

    # Enable mouse scroll for everything inside the frame
    def _on_mouse_wheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    scroll_frame.bind("<Enter>", lambda e: container.bind_all("<MouseWheel>", _on_mouse_wheel))
    scroll_frame.bind("<Leave>", lambda e: container.unbind_all("<MouseWheel>"))

    return container, scroll_frame
