import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)

# Global data
data_row_limit = 100
data = load_breast_cancer()
data_subset = data.data[:data_row_limit, :]
data.data = data.data[:data_row_limit, :]
data.target = data.target[:data_row_limit]
target_subset = data.target[:data_row_limit]

# Create the DataFrame
df = pd.DataFrame(data_subset, columns=data.feature_names)
df['target'] = target_subset

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
        ttk.Button(self, text="Cross-validation", command=self.run_cross_val_score).pack(pady=5)
        ttk.Button(self, text="Plot ROC Curve", command=self.plot_roc_curve).pack(pady=5)
        ttk.Button(self, text="Train Logistic with Regularization", command=self.train_logistic_with_regularization).pack(pady=5)
        ttk.Button(self, text="GaussianNB vs Logistic", command=self.compare_gaussian_nb_logistic).pack(pady=5)
        ttk.Button(self, text="SVC vs Logistic", command=self.compare_svc_logistic).pack(pady=5)
        ttk.Button(self, text="Decision Boundary Plot (2 features)", command=self.plot_decision_boundary).pack(pady=5)
        ttk.Button(self, text="RandomForest vs DecisionTree", command=self.compare_random_forest_decision_tree).pack(pady=5)
        ttk.Button(self, text="BaggingClassifier", command=self.run_bagging_classifier).pack(pady=5)
        ttk.Button(self, text="Show discussion: precision vs recall", command=self.discussion_precision_vs_recall).pack(pady=5)


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
        
        messagebox.showinfo("Linear Regression", f"predict mean radius (row 1-15):{formated_pred}")

    def run_logistic_regression(self):
        # Split data
        print(data.data)
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        model = LogisticRegression(max_iter=1000)
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

    
    def discussion_precision_vs_recall(self):
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

    def run_cross_val_score(self):
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, data.data, data.target, cv=5)
        messagebox.showinfo("Cross-validation Score", f"Cross-validation scores: {scores}\nMean score: {scores.mean()}")

    def plot_roc_curve(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def train_logistic_with_regularization(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Logistic model without regularization
        model_none = LogisticRegression(max_iter=1000, penalty=None, solver="lbfgs")
        model_none.fit(X_train, y_train)
        y_pred_none = model_none.predict(X_test)

        # Logistic model with L1 regularization
        model_l1 = LogisticRegression(max_iter=1000, penalty='l1', solver="liblinear")
        model_l1.fit(X_train, y_train)
        y_pred_l1 = model_l1.predict(X_test)

        # Logistic model with L2 regularization
        model_l2 = LogisticRegression(max_iter=1000, penalty='l2', solver="lbfgs")
        model_l2.fit(X_train, y_train)
        y_pred_l2 = model_l2.predict(X_test)

        acc_none, p_none, r_none, f1_none = self.calc_performence_scores(y_test, y_pred_none)
        acc_11, p_11, r_11, f1_11 = self.calc_performence_scores(y_test, y_pred_l1) 
        acc_12, p_12, r_12, f1_12 = self.calc_performence_scores(y_test, y_pred_l2)

        # Show results in a messagebox
        result_text = (
            f"Logistic Regression Penalties:\n"
            f"Penalties: None:\n"
            f"Accuracy: {acc_none:.2f}\n"
            f"Precision: {p_none:.2f}\n"
            f"Recall: {r_none:.2f}\n"
            f"F1-score: {f1_none:.2f}\n\n"
            f"Penalties: 11:\n"
            f"Accuracy: {acc_11:.2f}\n"
            f"Precision: {p_11:.2f}\n"
            f"Recall: {r_11:.2f}\n"
            f"F1-score: {f1_11:.2f}\n\n"
            f"Penalties: 12:\n"
            f"Accuracy: {acc_12:.2f}\n"
            f"Precision: {p_12:.2f}\n"
            f"Recall: {r_12:.2f}\n"
            f"F1-score: {f1_12:.2f}"
        )

        messagebox.showinfo("Logistic penalty", result_text)

    def compare_gaussian_nb_logistic(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Gaussian Naive Bayes
        model_gnb = GaussianNB()
        model_gnb.fit(X_train, y_train)
        y_pred_gnb = model_gnb.predict(X_test)

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        acc_gnb, p_gnb, r_gnb, f1_gnb = self.calc_performence_scores(y_test, y_pred_gnb)
        acc_lr, p_lr, r_lr, f1_lr = self.calc_performence_scores(y_test, y_pred_lr)

        result_text = (
            f"Gaussian Naive Bayes:\n"
            f"Accuracy: {acc_gnb:.2f}\n"
            f"Precision: {p_gnb:.2f}\n"
            f"Recall: {r_gnb:.2f}\n"
            f"F1-score: {f1_gnb:.2f}\n\n"
            f"Logistic Regression:\n"
            f"Accuracy: {acc_lr:.2f}\n"
            f"Precision: {p_lr:.2f}\n"
            f"Recall: {r_lr:.2f}\n"
            f"F1-score: {f1_lr:.2f}"
        )
        messagebox.showinfo("GaussianNB vs Logistic Regression", result_text)

    def compare_svc_logistic(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Support Vector Classifier
        model_svc = SVC(kernel='linear')
        model_svc.fit(X_train, y_train)
        y_pred_svc = model_svc.predict(X_test)

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        acc_svc, p_svc, r_svc, f1_svc = self.calc_performence_scores(y_test, y_pred_svc)
        acc_lr, p_lr, r_lr, f1_lr = self.calc_performence_scores(y_test, y_pred_lr)

        result_text = (
            f"SVC (linear kernel):\n"
            f"Accuracy: {acc_svc:.2f}\n"
            f"Precision: {p_svc:.2f}\n"
            f"Recall: {r_svc:.2f}\n"
            f"F1-score: {f1_svc:.2f}\n\n"
            f"Logistic Regression:\n"
            f"Accuracy: {acc_lr:.2f}\n"
            f"Precision: {p_lr:.2f}\n"
            f"Recall: {r_lr:.2f}\n"
            f"F1-score: {f1_lr:.2f}"
        )
        messagebox.showinfo("SVC vs Logistic Regression", result_text)

    def plot_decision_boundary(self):
        # Use only 2 features for decision boundary plot
        X = data.data[:, :2]
        y = data.target
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
        plt.xlabel('Mean radius')
        plt.ylabel('Mean texture')
        plt.title("Decision Boundary (Logistic Regression)")
        plt.show()

    def compare_random_forest_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Random Forest Classifier
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

        # Decision Tree Classifier
        model_dt = DecisionTreeClassifier()
        model_dt.fit(X_train, y_train)
        y_pred_dt = model_dt.predict(X_test)

        acc_rf, p_rf, r_rf, f1_rf = self.calc_performence_scores(y_test, y_pred_rf)
        acc_dt, p_dt, r_dt, f1_dt = self.calc_performence_scores(y_test, y_pred_dt)

        result_text = (
            f"Random Forest Classifier:\n"
            f"Accuracy: {acc_rf:.2f}\n"
            f"Precision: {p_rf:.2f}\n"
            f"Recall: {r_rf:.2f}\n"
            f"F1-score: {f1_rf:.2f}\n\n"
            f"Decision Tree Classifier:\n"
            f"Accuracy: {acc_dt:.2f}\n"
            f"Precision: {p_dt:.2f}\n"
            f"Recall: {r_dt:.2f}\n"
            f"F1-score: {f1_dt:.2f}"
        )
        messagebox.showinfo("RandomForest vs DecisionTree", result_text)

    def run_bagging_classifier(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        
        # Bagging Classifier
        model_bagging = BaggingClassifier(estimator=DecisionTreeClassifier())
        model_bagging.fit(X_train, y_train)
        y_pred_bagging = model_bagging.predict(X_test)

        acc_bagging, p_bagging, r_bagging, f1_bagging = self.calc_performence_scores(y_test, y_pred_bagging)
        self.show_results("BaggingClassifier", acc_bagging, p_bagging, r_bagging, f1_bagging)

# Run the app
if __name__ == "__main__":
    app = MLApp()
    app.mainloop()