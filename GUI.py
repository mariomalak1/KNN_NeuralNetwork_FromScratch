import tkinter as tk
from tkinter import filedialog, messagebox
import os

from preprocessing import preprocessing

class ANN_KNN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN & KNN Model Evaluation")

        # Create labels and input fields for the GUI
        self.create_widgets()

    def create_widgets(self):
        # Label for percentage of rows
        self.label_percentage = tk.Label(self.root, text="Percentage of Rows to Read (%):")
        self.label_percentage.grid(row=0, column=0, padx=10, pady=10)
        
        # Default value for percentage
        self.entry_percentage = tk.Entry(self.root)
        self.entry_percentage.insert(0, "80")  # Default percentage
        self.entry_percentage.grid(row=0, column=1, padx=10, pady=10)
        
        # Label for train data size
        self.label_train_size = tk.Label(self.root, text="Train Data Size (%):")
        self.label_train_size.grid(row=1, column=0, padx=10, pady=10)
        
        # Default value for train size
        self.entry_train_size = tk.Entry(self.root)
        self.entry_train_size.insert(0, "70")  # Default train size
        self.entry_train_size.grid(row=1, column=1, padx=10, pady=10)

        # Label for file location
        self.label_file_location = tk.Label(self.root, text="File Location (CSV/Excel):")
        self.label_file_location.grid(row=2, column=0, padx=10, pady=10)

        # Default value for file location (can be customized)
        self.entry_file_location = tk.Entry(self.root)
        self.entry_file_location.insert(0, "data.csv")  # Default file path
        self.entry_file_location.grid(row=2, column=1, padx=10, pady=10)
        
        # Browse Button for file location
        self.button_browse = tk.Button(self.root, text="Browse", command=self.browse_file)
        self.button_browse.grid(row=2, column=2, padx=10, pady=10)

        # Label for KNN's k value
        self.label_knn_k = tk.Label(self.root, text="KNN - k Value:")
        self.label_knn_k.grid(row=3, column=0, padx=10, pady=10)
        
        # Default value for KNN's k
        self.entry_knn_k = tk.Entry(self.root)
        self.entry_knn_k.insert(0, "5")  # Default k value
        self.entry_knn_k.grid(row=3, column=1, padx=10, pady=10)
        
        # Label for learning rate
        self.label_learning_rate = tk.Label(self.root, text="Learning Rate:")
        self.label_learning_rate.grid(row=4, column=0, padx=10, pady=10)
        
        # Default value for learning rate
        self.entry_learning_rate = tk.Entry(self.root)
        self.entry_learning_rate.insert(0, "0.05")  # Default learning rate
        self.entry_learning_rate.grid(row=4, column=1, padx=10, pady=10)
        
        # Button to run evaluation
        self.button_run = tk.Button(self.root, text="Run Evaluation", command=self.run_evaluation)
        self.button_run.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Output labels for KNN and ANN accuracy
        self.label_knn_result = tk.Label(self.root, text="KNN Accuracy:")
        self.label_knn_result.grid(row=6, column=0, padx=10, pady=10)
        
        self.label_ann_result = tk.Label(self.root, text="ANN Accuracy:")
        self.label_ann_result.grid(row=7, column=0, padx=10, pady=10)
        
        self.knn_accuracy = tk.Label(self.root, text="Not Evaluated")
        self.knn_accuracy.grid(row=6, column=1, padx=10, pady=10)
        
        self.ann_accuracy = tk.Label(self.root, text="Not Evaluated")
        self.ann_accuracy.grid(row=7, column=1, padx=10, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        self.entry_file_location.delete(0, tk.END)
        self.entry_file_location.insert(0, file_path)

    def run_evaluation(self):
        # Get user inputs
        try:
            percentage = float(self.entry_percentage.get())
            train_size = float(self.entry_train_size.get())
            file_location = self.entry_file_location.get()
            knn_k = int(self.entry_knn_k.get())
            learning_rate = float(self.entry_learning_rate.get())

            if not (0 < percentage <= 100 and 0 < train_size <= 100):
                raise ValueError("Percentage and Train Size should be between 0 and 100.")

            if not os.path.exists(file_location):
                raise ValueError("File path does not exist.")

            # Call preprocessing function for evaluation
            result = preprocessing(file_location, percentage, train_size, knn_k, learning_rate)

            if result:
                knn_acc, ann_acc = result
                self.knn_accuracy.config(text=f"{knn_acc * 100:.2f}%")
                self.ann_accuracy.config(text=f"{ann_acc * 100:.2f}%")
            else:
                messagebox.showerror("Error", "Failed to evaluate models.")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

# Initialize the Tkinter window
root = tk.Tk()
app = ANN_KNN_GUI(root)
root.mainloop()
