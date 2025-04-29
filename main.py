import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import discussions as discussion
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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
        self.geometry("1000x400")

        # Task label
        model_label = ttk.Label(self, text="Model Analysis", font=("Arial", 12, "bold"))
        model_label.grid(row=0, column=0, columnspan=5, pady=10)

        # Task buttons
        model_buttons = [
            ("Show Dataset", self.show_data),
            ("Run Linear Regression", self.run_linear_regression),
            ("Logistic Regression", self.run_logistic_regression),
            ("Decision Tree Classifier", self.run_decision_tree),
            ("Visualize Decision Tree", self.visualize_decision_tree),
            ("Show Confusion Matrix", self.show_conf_matrix),
            ("Compare Models", self.compare_logistic_regression_decision_tree),
            ("Cross-validation", self.run_cross_val_score),
            ("Plot ROC Curve", self.plot_roc_curve),
            ("Train Logistic with Regularization", self.train_logistic_with_regularization),
            ("GaussianNB vs Logistic", self.compare_gaussian_nb_logistic),
            ("SVC vs Logistic", self.compare_svc_logistic),
            ("Decision Boundary Plot", self.plot_decision_boundary),
            ("RandomForest vs DecisionTree", self.compare_random_forest_decision_tree),
            ("BaggingClassifier", self.run_bagging_classifier),
            ("Learning curve (decision tree)", self.learning_curve_decision_tree),
            ("Learning curve (random forest)", self.learning_curve_random_forest),
            ("Precision vs Recall Discussion", discussion.discussion_precision_vs_recall),
            ("Bias vs Variance decision tree Discussion", discussion.discussion_bias_variance_trees),
            ("Reduce Dimension with PCA", self.reduce_dimension_with_PCA),
        ]
        for idx, (text, command) in enumerate(model_buttons):
            row = idx // 5 + 1
            column = idx % 5
            ttk.Button(self, text=text, command=command).grid(row=row, column=column, pady=5, padx=5, sticky="ew")

        # Discussion label
        discussion_label = ttk.Label(self, text="Discussions", font=("Arial", 12, "bold"))
        discussion_label.grid(row=len(model_buttons) // 5 + 2, column=0, columnspan=5, pady=10)

        # Discussion buttons
        discussion_buttons = [
            ("Discussion Precision vs Recall", discussion.discussion_precision_vs_recall),
            ("Discussion Bias vs Variance", discussion.discussion_bias_variance_trees),
        ]
        for idx, (text, command) in enumerate(discussion_buttons):
            row = (len(model_buttons) // 5 + 3) + idx // 5
            column = idx % 5
            ttk.Button(self, text=text, command=command).grid(row=row, column=column, pady=5, padx=5, sticky="ew")

        

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

    def learning_curve_decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        
        model_dt = DecisionTreeClassifier(max_depth=1)

        model_dt.fit(X_train, y_train)

        train_sizes, train_scores, test_scores = learning_curve(model_dt, X_train, y_train)

        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score (test scores)')
        plt.title('Learning Curve (Decision Tree)')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

    def learning_curve_random_forest(self):
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        model_rf = RandomForestClassifier(max_depth=1)

        model_rf.fit(X_train, y_train)

        train_sizes, train_scores, test_scores = learning_curve(model_rf, X_train, y_train)

        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score (test scores)')
        plt.title('Learning Curve (Decision Tree)')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

    def reduce_dimension_with_PCA(self):
        from sklearn.decomposition import PCA
        X_reduced = PCA(n_components=10).fit_transform(data.data)
        messagebox.showinfo("PCA Reduction", f"Reduced dimensions from {data.data.shape[1]} to {X_reduced.shape[1]}")
        return X_reduced

# Run the app
if __name__ == "__main__":
    app = MLApp()
    app.mainloop()