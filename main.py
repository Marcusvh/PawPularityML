import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# Global data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# App GUI
class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML GUI – Breast Cancer Analysis")
        self.geometry("500x400")

        ttk.Button(self, text="Show Dataset", command=self.show_data).pack(pady=5)
        ttk.Button(self, text="Run Linear Regression", command=self.run_linear_regression).pack(pady=5)
        ttk.Button(self, text="Logistic Regression", command=self.run_logistic_regression).pack(pady=5)
        ttk.Button(self, text="Decision Tree Classifier", command=self.run_decision_tree).pack(pady=5)
        ttk.Button(self, text="Visualize Decision Tree", command=self.visualize_decision_tree).pack(pady=5)
        ttk.Button(self, text="Show Confusion Matrix", command=self.show_conf_matrix).pack(pady=5)
        ttk.Button(self, text="Compare Models (LogisticsRegression and Decision tree)", command=self.compare_logistic_regression_decision_tree).pack(pady=5)
        ttk.Button(self, text="Show discussion", command=self.discussion).pack(pady=5)

    def show_data(self):
        top = tk.Toplevel(self)
        top.title("Dataset")
        frame = ttk.Frame(top)
        frame.pack(expand=True, fill="both")

        # Treeview
        tree = ttk.Treeview(frame, columns=list(df.columns), show="headings")
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        for _, row in df.head(50).iterrows():
            tree.insert("", "end", values=list(row))

        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Layout
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def run_linear_regression(self):
        X = data.data[:15, :]
        y = data.data[:15, 0]  # mean radius
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X[:15])
        formated_pred = [f"{pred[i]:.2f}\n" for i in range(len(pred))]
        
        messagebox.showinfo("Linear Regression", f"predict mean radius (row 1-15):\n{formated_pred}")

    def run_logistic_regression(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        model = LogisticRegression(max_iter=1000, solver="saga")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc, p, r, f1 = self.calc_performence_scores(y_test, y_pred)

        self.y_test = y_test
        self.y_pred = y_pred
        self.show_results("Logistic Regression", acc, p, r, f1)

        cm = confusion_matrix(y_test, y_pred)
        self.conf_matrix = cm  # Gives data to show in confusion matrix

    def run_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Decision Tree Classifier model
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics for Decision Tree
        acc, p, r, f1 = self.calc_performence_scores(y_test, y_pred)

        self.show_results("Decision Tree", acc, p, r, f1)
    
    def visualize_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Decision Tree Classifier model
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
    
        plt.figure(figsize=(12, 6))
        plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
        plt.title("Decision Tree")
        plt.show()

    def show_results(self, model_name, acc, p, r, f1):
        # Show the results of the model
        messagebox.showinfo(model_name, f"Accuracy: {acc:.2f}\nPrecision: {p:.2f}\nRecall: {r:.2f}\nF1-score: {f1:.2f}")

    def show_conf_matrix(self):
        if hasattr(self, 'conf_matrix'):
            import numpy as np

            labels = np.array([["True negative (TN)", "False positive (FP)"], ["False negative (FN)", "True positive (TP)"]])
            values = self.conf_matrix
            annot = np.empty_like(values, dtype=object)

            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    annot[i, j] = f"{labels[i, j]}\n{values[i, j]}"

            sns.heatmap(values, annot=annot, fmt='', cmap='Blues', cbar=False,
                        xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.title("Confusion Matrix (Logistic Regression)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
        else:
            messagebox.showerror("Error", "Run Logistic Regression first.")

    def show_metrics(self):
        if hasattr(self, 'y_test'):
            p = precision_score(self.y_test, self.y_pred)
            r = recall_score(self.y_test, self.y_pred)
            f1 = f1_score(self.y_test, self.y_pred)
            messagebox.showinfo("Metrics", f"Precision: {p:.2f}\nRecall: {r:.2f}\nF1-score: {f1:.2f}")
        else:
            messagebox.showerror("Error", "Run Logistic Regression first.")

    def compare_logistic_regression_decision_tree(self):
        # Compare Logistic Regression and Decision Tree Classifier
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        # Decision Tree Classifier
        model_dt = DecisionTreeClassifier(max_depth=3)
        model_dt.fit(X_train, y_train)
        y_pred_dt = model_dt.predict(X_test)

        acc_lr, p_lr, r_lr, f1_lr = self.calc_performence_scores(y_test, y_pred_lr) # Logistic Regression
        acc_dt, p_dt, r_dt, f1_dt = self.calc_performence_scores(y_test, y_pred_dt) # Decision Tree

        # Show results in a messagebox
        result_text = (
            f"Logistic Regression:\n"
            f"Accuracy: {acc_lr:.2f}\n"
            f"Precision: {p_lr:.2f}\n"
            f"Recall: {r_lr:.2f}\n"
            f"F1-score: {f1_lr:.2f}\n\n"
            f"Decision Tree Classifier:\n"
            f"Accuracy: {acc_dt:.2f}\n"
            f"Precision: {p_dt:.2f}\n"
            f"Recall: {r_dt:.2f}\n"
            f"F1-score: {f1_dt:.2f}"
        )
        
        messagebox.showinfo("Model Comparison", result_text)
    
    def calc_performence_scores(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        r = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return acc, p, r, f1
    
    def discussion(self):
        # Discussion about the models
        discussion_text = (
            "Precision vs Recall\n"
            "Precision er forholdet mellem sande positive forudsigelser og det samlede antal forudsagte positive.\n"
            "Formel for Precision: Precision = TP / (TP + FP)\n\n"
            "Recall er forholdet mellem sande positive forudsigelser og det samlede antal faktiske positive.\n"
            "Formel for Recall: Recall = TP / (TP + FN)\n\n"
            "Hvornår man bør prioritere Precision frem for Recall:\n"
            "Hvis omkostningen ved en falsk positiv er høj, bør man prioritere Precision. For eksempel i spamfiltrering: at markere en legitim e-mail som spam (falsk positiv) kan være til stor gene for brugeren.\n\n"
            "Hvornår man bør prioritere Recall frem for Precision:\n"
            "Hvis omkostningen ved en falsk negativ er høj, bør man prioritere Recall. For eksempel i medicinske diagnoser: at undlade at opdage en sygdom (falsk negativ) kan have alvorlige konsekvenser for patienten."
        )
        messagebox.showinfo("Discussion", discussion_text)
# Run the app
if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
