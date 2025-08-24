import tkinter as tk

class HUD(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mindful Focus")
        self.geometry("420x180")
        self.configure(bg="#0f172a")
        self.resizable(False, False)

        self.lbl_title = tk.Label(self, text="Mindful Focus", fg="#e2e8f0", bg="#0f172a", font=("Arial", 16, "bold"))
        self.lbl_title.pack(pady=(14,6))

        row = tk.Frame(self, bg="#0f172a")
        row.pack(pady=4)
        self.lbl_bpm = tk.Label(row, text="Blink/min: --", fg="#cbd5e1", bg="#0f172a", font=("Arial", 12))
        self.lbl_bpm.pack(side=tk.LEFT, padx=10)
        self.lbl_iris = tk.Label(row, text="Iris ratio: --", fg="#cbd5e1", bg="#0f172a", font=("Arial", 12))
        self.lbl_iris.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(self, text="Status: calibrating…", fg="#0f172a", bg="#fbbf24", font=("Arial", 13, "bold"), width=30)
        self.lbl_status.pack(pady=12)

    def set_calibrating(self):
        self.lbl_status.config(text="Status: Calibrating…", bg="#fbbf24", fg="#0f172a")

    def set_status(self, bmp: float, iris: float, tag: str):
        self.lbl_bpm.config(text=f"Blink/min: {bmp:0.2f}")
        self.lbl_iris.config(text=f"Iris ratio: {iris:0.4f}")
        if tag == "OK":
            self.lbl_status.config(text="Status: OK", bg="#22c55e", fg="white")
        elif tag == "Focus ↑":
            self.lbl_status.config(text="Status: Focus ↑", bg="#60a5fa", fg="#0b1220")
        elif tag == "Eye Strain ↑":
            self.lbl_status.config(text="Status: Eye Strain ↑", bg="#ef4444", fg="white")
        else:
            self.lbl_status.config(text=f"Status: {tag}", bg="#94a3b8", fg="#0b1220")
