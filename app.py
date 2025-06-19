import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
import datetime
import torch
import os
from cutting_fruit import detect_and_crop  # Hàm cắt ảnh từ YOLO


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Xử lí ảnh nhóm 2")
        self.root.geometry("900x564")
        self.root.resizable(False, False)
        self.root.iconbitmap("logo.ico")

        # Chuyển sang giao diện sáng
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")  # Có thể đổi theme nếu muốn

        # Load mô hình YOLO
        self.model = YOLO("model/best_205_epoch.pt")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Biến kiểm soát việc stream camera
        self.streaming = True

        # ===== FRAME CHÍNH =====
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ===== FRAME VIDEO =====
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(fill="both", expand=True)

        # ===== NÚT BÊN DƯỚI PHẢI =====
        button_style = {
            "font": ("Roboto", 14, "bold"),
            "corner_radius": 8,
            "text_color": "white",
            "width": 128,
            "height": 48,
        }

        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame.pack(side="bottom", anchor="se", padx=10, pady=10)

        self.capture_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Nhận diện độ chín",
            command=self.capture_image,
            fg_color="#2196F3",
            hover_color="#1976D2",
            **button_style
        )
        self.capture_btn.pack(side="right", padx=5)

        self.quit_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Thoát",
            command=self.quit_app,
            fg_color="#FF5252",
            hover_color="#D32F2F",
            **button_style
        )
        self.quit_btn.pack(side="right", padx=5)

        # Mở webcam
        self.cap = cv2.VideoCapture(0)

        # Bắt đầu hiển thị video
        self.update_frame()

    # ===== Cập nhật khung hình từ webcam =====
    def update_frame(self):
        if not self.streaming:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (880, 480))
            results = self.model.predict(frame, device="cuda" if torch.cuda.is_available() else "cpu")
            new_frame = results[0].plot()
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(new_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(200, self.update_frame)

    # ===== Chụp ảnh và hiển thị kết quả =====
    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.streaming = False

        fruit_list = detect_and_crop(frame, self.model)

        self.video_label.pack_forget()
        self.buttons_frame.pack_forget()

        self.result_frame = ctk.CTkFrame(self.main_frame, fg_color="#f5f5f5")
        self.result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Nút quay lại
        back_btn = ctk.CTkButton(self.result_frame, text="← Quay lại", command=self.back_to_main)
        back_btn.pack(anchor="nw", padx=10, pady=10)

        # ============ SCROLL FRAME ============
        canvas = ctk.CTkCanvas(self.result_frame, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(self.result_frame, orientation="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Frame chứa nội dung cuộn
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="#f5f5f5")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # ============ GRID ẢNH ============
        max_columns = 5  # ảnh trên 1 hàng

        for i, (name, img) in enumerate(fruit_list):
            row = i // max_columns
            col = i % max_columns

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((150, 150))
            tk_img = ImageTk.PhotoImage(pil_img)

            item_frame = ctk.CTkFrame(scrollable_frame, fg_color="white", corner_radius=10)
            item_frame.grid(row=row, column=col, padx=10, pady=10)

            img_label = ctk.CTkLabel(item_frame, image=tk_img, text="")
            img_label.image = tk_img
            img_label.pack(pady=(10, 0))

            info = f"Tên: {name}\nĐộ chín: Chín"
            info_label = ctk.CTkLabel(item_frame, text=info, text_color="black", justify="center")
            info_label.pack(pady=(5, 10))


    # ===== Trở lại màn hình chính =====
    def back_to_main(self):
        self.result_frame.destroy()
        self.video_label.pack(fill="both", expand=True)
        self.buttons_frame.pack(side="bottom", anchor="se", padx=10, pady=10)
        self.streaming = True
        self.update_frame()

    # ===== Thoát chương trình =====
    def quit_app(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = CameraApp(ctk.CTk())
    app.root.mainloop()
