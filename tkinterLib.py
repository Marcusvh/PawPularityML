import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def displayRegressionErrors(frame, tk, mae, mse, rmse, r2):
    title_label = tk.Label(frame, text="Model Performance Metrics", font=("Arial", 14, "bold"))
    title_label.grid(column=0, row=5, columnspan=5, pady=10)

    linearText = tk.Label(frame, text="Errors when using Regression", font=("Arial", 12, "italic"))
    linearText.grid(column=0, row=6, columnspan=5, pady=5)

    r2Score = tk.Label(frame, text=f"R² Score: {r2:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    r2Score.grid(column=0, row=7, sticky="w", pady=5)

    rmseScore = tk.Label(frame, text=f"Root Mean Squared Error (RMSE): {rmse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    rmseScore.grid(column=0, row=8, sticky="w", pady=5)

    mseScore = tk.Label(frame, text=f"Mean Squared Error (MSE): {mse:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    mseScore.grid(column=0, row=9, sticky="w", pady=5)

    maeScore = tk.Label(frame, text=f"Mean Absolute Error (MAE): {mae:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
    maeScore.grid(column=0, row=10, sticky="w", pady=5)

    # extra config
    r2Score.config(bg="lightblue")
    rmseScore.config(bg="lightyellow")
    mseScore.config(bg="lightgreen")
    maeScore.config(bg="lightcoral")

def headerDisplay(frame, tk, browseFiles, window):
    # File Explorer label
    label_file_explorer = tk.Label(frame, text="File Explorer using Tkinter", width=100, height=2, fg="blue")
    label_file_explorer.grid(column=0, row=1, columnspan=5, pady=10)

    button_explore = tk.Button(frame, text="Browse Files", command=browseFiles)
    button_explore.grid(column=0, row=2, columnspan=5, pady=10)

    button_exit = tk.Button(frame, text="Exit", command=window.quit)
    button_exit.grid(column=0, row=3, columnspan=5, pady=10)

    return label_file_explorer

def displayCrossValidationScores(frame, tk, cv_scores):
    title_label = tk.Label(frame, text="Cross-Validation Scores", font=("Arial", 14, "bold"))
    title_label.grid(column=0, row=11, columnspan=5, pady=10)

    for idx, score in enumerate(cv_scores):
        score_label = tk.Label(frame, text=f"Fold {idx + 1}: {score:.5f}", font=("Arial", 10, "bold"), anchor="w", padx=10)
        score_label.grid(column=0, row=12 + idx, sticky="w", pady=5)

def displayROCCurve(frame, fpr, tpr, roc_auc):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(column=0, row=20, columnspan=5, pady=10)