def displayLinearRegressionErrors(linear_frame, tk, mae, mse, rmse, r2):
    title_label = tk.Label(linear_frame, text="Model Performance Metrics", font=("Arial", 14, "bold"))
    title_label.grid(column=0, row=5, columnspan=5, pady=10)

    linearText = tk.Label(linear_frame, text="Errors when using LinearRegression", font=("Arial", 12, "italic"))
    linearText.grid(column=0, row=6, columnspan=5, pady=5)

    r2Score = tk.Label(linear_frame, text=f"R² Score: {r2:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    r2Score.grid(column=0, row=7, sticky="w", pady=5)

    rmseScore = tk.Label(linear_frame, text=f"Root Mean Squared Error (RMSE): {rmse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    rmseScore.grid(column=0, row=8, sticky="w", pady=5)

    mseScore = tk.Label(linear_frame, text=f"Mean Squared Error (MSE): {mse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    mseScore.grid(column=0, row=9, sticky="w", pady=5)

    maeScore = tk.Label(linear_frame, text=f"Mean Absolute Error (MAE): {mae:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    maeScore.grid(column=0, row=10, sticky="w", pady=5)

    # extra config
    r2Score.config(bg="lightblue")
    rmseScore.config(bg="lightyellow")
    mseScore.config(bg="lightgreen")
    maeScore.config(bg="lightcoral")

def displayLogisticRegressionErrors(logistic_frame, tk, accuracy):
    title_label = tk.Label(logistic_frame, text="Model Performance Metrics", font=("Arial", 14, "bold"))
    title_label.grid(column=0, row=5, columnspan=5, pady=10)

    logisticText = tk.Label(logistic_frame, text="Performence when using LogisticRegression", font=("Arial", 12, "italic"))
    logisticText.grid(column=0, row=6, columnspan=5, pady=5)

    accuracyScore = tk.Label(logistic_frame, text=f"Accuracy score: {accuracy:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    accuracyScore.grid(column=0, row=7, sticky="w", pady=5)

    # extra config
    accuracyScore.config(bg="lightblue")

def headerDisplay(frame, tk, browseFiles, window):
    # File Explorer label
    label_file_explorer = tk.Label(frame, text="File Explorer using Tkinter", width=100, height=2, fg="blue")
    label_file_explorer.grid(column=0, row=1, columnspan=5, pady=10)

    button_explore = tk.Button(frame, text="Browse Files", command=browseFiles)
    button_explore.grid(column=0, row=2, columnspan=5, pady=10)

    button_exit = tk.Button(frame, text="Exit", command=window.quit)
    button_exit.grid(column=0, row=3, columnspan=5, pady=10)

    return label_file_explorer