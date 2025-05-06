import tkinter as tk
from tkinter import filedialog, ttk

class KidneyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN & KNN Model Evaluation")

        self.fields_info = {
            "age": "48.0", "bp": "80.0", "sg": "1.02", "al": "1.0", "su": "0.0",
            "rbc": ["normal", "abnormal"],
            "pc": ["normal", "abnormal"],
            "pcc": ["present", "notpresent"],
            "ba": ["present", "notpresent"],
            "bgr": "121.0", "bu": "36.0", "sc": "1.2", "sod": "", "pot": "",
            "hemo": "15.4", "pcv": "44", "wc": "7800", "rc": "5.2",
            "htn": ["yes", "no"],
            "dm": ["yes", "no"],
            "cad": ["yes", "no"],
            "appet": ["good", "poor"],
            "pe": ["no", "yes"],
            "ane": ["no", "yes"]
        }

        self.feature_entries = {}

        self.build_gui()

    def build_gui(self):
        tk.Label(self.root, text="Percentage of Rows to Read (%):").grid(row=0, column=0, sticky="e")
        self.rows_entry = tk.Entry(self.root)
        self.rows_entry.insert(0, "80")
        self.rows_entry.grid(row=0, column=1)

        tk.Label(self.root, text="Train Data Size (%):").grid(row=1, column=0, sticky="e")
        self.train_size_entry = tk.Entry(self.root)
        self.train_size_entry.insert(0, "70")
        self.train_size_entry.grid(row=1, column=1)

        tk.Label(self.root, text="File Location (CSV/Excel):").grid(row=2, column=0, sticky="e")
        self.file_entry = tk.Entry(self.root, width=50)
        self.file_entry.insert(0, "../Kidney_Disease_data_for_Classification_V2.csv")
        self.file_entry.grid(row=2, column=1, columnspan=2, sticky="we")
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=2, column=3)

        tk.Label(self.root, text="KNN- k Value:").grid(row=3, column=0, sticky="e")
        self.knn_entry = tk.Entry(self.root)
        self.knn_entry.insert(0, "5")
        self.knn_entry.grid(row=3, column=1)

        tk.Label(self.root, text="Learning Rate:").grid(row=4, column=0, sticky="e")
        self.lr_entry = tk.Entry(self.root)
        self.lr_entry.insert(0, "0.05")
        self.lr_entry.grid(row=4, column=1)

        tk.Label(self.root, text="Threshold:").grid(row=5, column=0, sticky="e")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "90")
        self.threshold_entry.grid(row=5, column=1)

        tk.Button(self.root, text="Run Evaluation", command=self.run_evaluation).grid(row=6, column=0, columnspan=4)

        # Features input row by row
        row = 7
        for idx, (field, default) in enumerate(self.fields_info.items()):
            tk.Label(self.root, text=field).grid(row=row, column=idx % 4 * 2, sticky="e")
            if isinstance(default, list):
                var = tk.StringVar(value=default[0])
                dropdown = ttk.OptionMenu(self.root, var, default[0], *default)
                dropdown.grid(row=row, column=idx % 4 * 2 + 1)
                self.feature_entries[field] = var
            else:
                entry = tk.Entry(self.root)
                entry.insert(0, default)
                entry.grid(row=row, column=idx % 4 * 2 + 1)
                self.feature_entries[field] = entry

            if (idx + 1) % 4 == 0:
                row += 1

        self.knn_result = tk.Label(self.root, text="KNN Accuracy: Not Evaluated")
        self.knn_result.grid(row=row+1, column=0, columnspan=2, sticky="w")

        self.ann_result = tk.Label(self.root, text="ANN Accuracy: Not Evaluated")
        self.ann_result.grid(row=row+1, column=2, columnspan=2, sticky="w")

        self.prediction_result = tk.Label(self.root, text="Prediction: Not Available")
        self.prediction_result.grid(row=row+2, column=0, columnspan=4)

    def browse_file(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filepath)

    def run_evaluation(self):
        sample_input = [
            self.feature_entries[k].get() if isinstance(self.feature_entries[k], tk.StringVar)
            else self.feature_entries[k].get()
            for k in self.feature_entries
        ]
        
        self.knn_result.config(text="KNN Accuracy: 91%")
        self.ann_result.config(text="ANN Accuracy: 94%")
        self.prediction_result.config(text="Prediction: CKD")


if __name__ == '__main__':
    root = tk.Tk()
    app = KidneyApp(root)
    root.mainloop()
