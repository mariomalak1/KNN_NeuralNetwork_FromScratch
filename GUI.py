import tkinter as tk
from tkinter import filedialog, messagebox

import preprocessing


FIELDS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc",
    "rc", "htn", "dm", "cad", "appet", "pe", "ane"
]

class ANN_KNN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN & KNN Model Evaluation")
        self.feature_entries = {}
        self.create_widgets()

    def create_widgets(self):
        # First Row
        tk.Label(self.root, text="Percentage of Rows to Read (%):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.percentage_entry = tk.Entry(self.root, width=5)
        self.percentage_entry.insert(0, "80")
        self.percentage_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.root, text="Train Data Size (%):").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.train_entry = tk.Entry(self.root, width=5)
        self.train_entry.insert(0, "70")
        self.train_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=5, padx=5, pady=5)

        # Second Row – File location
        tk.Label(self.root, text="File Location (CSV/Excel):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.file_entry = tk.Entry(self.root, width=60)
        self.file_entry.insert(0, "../Kidney_Disease_data_for_Classification_V2.csv")
        self.file_entry.grid(row=1, column=1, columnspan=5, padx=5, pady=5, sticky='w')

        # Third Row – Model parameters
        tk.Label(self.root, text="KNN– k Value:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.k_entry = tk.Entry(self.root, width=5)
        self.k_entry.insert(0, "5")
        self.k_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        tk.Label(self.root, text="Learning Rate:").grid(row=2, column=2, padx=5, pady=5, sticky='e')
        self.lr_entry = tk.Entry(self.root, width=5)
        self.lr_entry.insert(0, "0.05")
        self.lr_entry.grid(row=2, column=3, padx=5, pady=5, sticky='w')

        tk.Label(self.root, text="Threshold:").grid(row=2, column=4, padx=5, pady=5, sticky='e')
        self.thresh_entry = tk.Entry(self.root, width=5)
        self.thresh_entry.insert(0, "90")
        self.thresh_entry.grid(row=2, column=5, padx=5, pady=5, sticky='w')

        # Feature Inputs – 6 rows x 4 columns
        row_base = 3
        for i, field in enumerate(FIELDS):
            r = row_base + i // 4
            c = (i % 4) * 2
            tk.Label(self.root, text=field).grid(row=r, column=c, padx=5, pady=2, sticky='e')
            entry = tk.Entry(self.root, width=10)
            entry.grid(row=r, column=c + 1, padx=5, pady=2, sticky='w')
            self.feature_entries[field] = entry

        # Run Evaluation Button
        btn_row = row_base + len(FIELDS) // 4 + 1
        tk.Button(self.root, text="Run Evaluation", command=self.run_evaluation).grid(row=btn_row, column=0, columnspan=6, pady=10)

        # Output Labels
        tk.Label(self.root, text="KNN Accuracy:").grid(row=btn_row+1, column=0, sticky='e')
        self.knn_result = tk.Label(self.root, text="Not Evaluated")
        self.knn_result.grid(row=btn_row+1, column=1, sticky='w')

        tk.Label(self.root, text="ANN Accuracy:").grid(row=btn_row+1, column=2, sticky='e')
        self.ann_result = tk.Label(self.root, text="Not Evaluated")
        self.ann_result.grid(row=btn_row+1, column=3, sticky='w')

        tk.Label(self.root, text="Prediction:").grid(row=btn_row+1, column=4, sticky='e')
        self.prediction_result = tk.Label(self.root, text="Not Predicted")
        self.prediction_result.grid(row=btn_row+1, column=5, sticky='w')

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def run_evaluation(self):
        try:
            percentage = float(self.percentage_entry.get())
            train_size = float(self.train_entry.get())
            k = int(self.k_entry.get())
            lr = float(self.lr_entry.get())
            threshold = float(self.thresh_entry.get())
            file_path = self.file_entry.get()
            sample_input = [entry.get() for entry in self.feature_entries.values()]

            # Dummy values (replace this with your actual model call)
            knn_acc, ann_acc, prediction = preprocessing.preprocessing(file_path, percentage, train_size, k, lr, 3, threshold, sample_input)

            # Display results
            self.knn_result.config(text=f"{knn_acc*100:.2f}%")
            self.ann_result.config(text=f"{ann_acc*100:.2f}%")
            self.prediction_result.config(text=prediction)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Launch the app
root = tk.Tk()
app = ANN_KNN_GUI(root)
root.mainloop()
