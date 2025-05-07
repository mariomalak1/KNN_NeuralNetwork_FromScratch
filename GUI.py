import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from preprocessing import preprocessing

class ANN_KNN_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN & KNN Model Evaluation")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.main_tab = ttk.Frame(self.notebook)
        self.knn_tab = ttk.Frame(self.notebook)
        self.ann_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.knn_tab, text="KNN Testing Data")
        self.notebook.add(self.ann_tab, text="ANN Testing Data")

        self.create_main_widgets()
        self.create_table(self.knn_tab, is_knn=True)
        self.create_table(self.ann_tab, is_knn=False)

    def create_main_widgets(self):
        self.label_percentage = tk.Label(self.main_tab, text="Percentage of Rows to Read (%):")
        self.label_percentage.grid(row=0, column=0, padx=10, pady=10)
        
        self.entry_percentage = tk.Entry(self.main_tab)
        self.entry_percentage.insert(0, "100")
        self.entry_percentage.grid(row=0, column=1, padx=10, pady=10)
        
        self.label_train_size = tk.Label(self.main_tab, text="Train Data Size (%):")
        self.label_train_size.grid(row=1, column=0, padx=10, pady=10)
        
        self.entry_train_size = tk.Entry(self.main_tab)
        self.entry_train_size.insert(0, "70")
        self.entry_train_size.grid(row=1, column=1, padx=10, pady=10)

        self.label_file_location = tk.Label(self.main_tab, text="File Location (CSV/Excel):")
        self.label_file_location.grid(row=2, column=0, padx=10, pady=10)

        self.entry_file_location = tk.Entry(self.main_tab)
        self.entry_file_location.insert(0, "../Kidney_Disease_data_for_Classification_V2.csv")
        self.entry_file_location.grid(row=2, column=1, padx=10, pady=10)
        
        self.button_browse = tk.Button(self.main_tab, text="Browse", command=self.browse_file)
        self.button_browse.grid(row=2, column=2, padx=10, pady=10)

        self.label_knn_k = tk.Label(self.main_tab, text="KNN - k Value:")
        self.label_knn_k.grid(row=3, column=0, padx=10, pady=10)
        
        self.entry_knn_k = tk.Entry(self.main_tab)
        self.entry_knn_k.insert(0, "3")
        self.entry_knn_k.grid(row=3, column=1, padx=10, pady=10)
        
        self.label_learning_rate = tk.Label(self.main_tab, text="Learning Rate:")
        self.label_learning_rate.grid(row=4, column=0, padx=10, pady=10)
        
        self.entry_learning_rate = tk.Entry(self.main_tab)
        self.entry_learning_rate.insert(0, "0.05")
        self.entry_learning_rate.grid(row=4, column=1, padx=10, pady=10)
        
        self.button_run = tk.Button(self.main_tab, text="Run Evaluation", command=self.run_evaluation)
        self.button_run.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.label_knn_result = tk.Label(self.main_tab, text="KNN Accuracy:")
        self.label_knn_result.grid(row=6, column=0, padx=10, pady=10)
        
        self.label_ann_result = tk.Label(self.main_tab, text="ANN Accuracy:")
        self.label_ann_result.grid(row=7, column=0, padx=10, pady=10)
        
        self.knn_accuracy = tk.Label(self.main_tab, text="Not Evaluated")
        self.knn_accuracy.grid(row=6, column=1, padx=10, pady=10)
        
        self.ann_accuracy = tk.Label(self.main_tab, text="Not Evaluated")
        self.ann_accuracy.grid(row=7, column=1, padx=10, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        self.entry_file_location.delete(0, tk.END)
        self.entry_file_location.insert(0, file_path)

    def run_evaluation(self):
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

            result = preprocessing(file_location, percentage, train_size, knn_k, learning_rate)

            if result:
                knn_acc, ann_acc, knn_records, ann_records = result
                self.knn_accuracy.config(text=f"{knn_acc * 100:.2f}%")
                self.ann_accuracy.config(text=f"{ann_acc * 100:.2f}%")
                self.populate_table(self.knn_table, knn_records)
                self.populate_table(self.ann_table, ann_records)
            else:
                messagebox.showerror("Error", "Failed to evaluate models.")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def create_table(self, tab, is_knn=True):
        # Create a container frame
        container = ttk.Frame(tab)
        container.pack(fill='both', expand=True)
        
        # Create the treeview
        tree = ttk.Treeview(container, show='headings')
        
        # Create scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        
        # Configure the treeview
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Use grid for better control
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        # Make sure the treeview can be scrolled with mouse wheel
        def _on_mousewheel(event):
            tree.yview_scroll(int(-1*(event.delta/120)), "units")
        
        tree.bind("<MouseWheel>", _on_mousewheel)
        
        if is_knn:
            self.knn_table = tree
        else:
            self.ann_table = tree

    def populate_table(self, tree, records):
        tree.delete(*tree.get_children())
        if not records:
            return
        
        # get the actual data labels from the first record, then delete it to not be displayed again
        headers = records[0]
        records.pop(0)

        tree["columns"] = headers
        for col in headers:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')

        # Insert the data rows
        for row in records:
            if isinstance(row, dict):
                values = list(row.values())
            else:
                values = row
            tree.insert("", "end", values=values)


# Run the GUI
root = tk.Tk()
app = ANN_KNN_GUI(root)
root.mainloop()
