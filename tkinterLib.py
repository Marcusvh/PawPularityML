def BrowseFilesDisplayErros(linear_frame, tk, mae, mse, rmse, r2):
    # adding labels to show the result.
        title_label = tk.Label(linear_frame, text="Model Performance Metrics", font=("Arial", 14, "bold"))
        title_label.grid(column=0, row=5, columnspan=5, pady=10)

        LinearText = tk.Label(linear_frame, text="Errors when using LinearRegression", font=("Arial", 12, "italic"))
        LinearText.grid(column=0, row=6, columnspan=5, pady=5)

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
