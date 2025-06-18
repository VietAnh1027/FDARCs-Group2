import customtkinter as ctk
from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
import datetime

class CameraApp:
    def __init__(self, root):
        self.root = root
        root.iconbitmap("logo.ico")
        self.root.title("Xử lí ảnh nhóm 2")
        self.root.geometry("900x564")
        self.root.resizable(False, False)
        self.model = YOLO("best.pt")

        # Giao diện
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Frame chính
        self.main_frame = ctk.CTkFrame(root, fg_color="#000000")
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Frame video
        self.video_frame = ctk.CTkFrame(self.main_frame, fg_color="#000000", corner_radius=20)
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Hiển thị video
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(fill="both", expand=True)

        # Control panel (xuống dưới video)
        self.control_frame = ctk.CTkFrame(self.main_frame, fg_color="#000000", corner_radius=10)
        self.control_frame.pack(fill="x", pady=(0, 10), padx=5)

        # Label trạng thái
        self.status_label = ctk.CTkLabel(
            self.control_frame,
            text="Ready",
            font=("Roboto", 14),
            text_color="#4CAF50"
        )
        self.status_label.pack(side="left", padx=5, pady=5)

        # Buttons style
        button_style = {
            "font": ("Roboto", 14, "bold"),
            "corner_radius": 8,
            "text_color": "white",
            "width": 128,
            "height": 64,
            "border_width": 0,
        }

        # Buttons
        self.buttons_frame = ctk.CTkFrame(self.control_frame, fg_color="#000000")
        self.buttons_frame.pack(side="right")

        self.tracking_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Nhận diện hoa quả",
            command=self.tracking,
            fg_color="#2196F3",
            hover_color="#1976D2",
            **button_style
        )
        self.tracking_btn.pack(side="right", padx=5)

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

        # Bắt đầu luồng camera
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (880, 480))  # Giảm chiều cao để vừa giao diện
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.status_label.configure(text="Streaming...")

        self.root.after(15, self.update_frame)

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.status_label.configure(text=f"Saved: {filename}", text_color="#4CAF50")
            print(f"📸 Đã lưu ảnh: {filename}")
        else:
            self.status_label.configure(text="Capture failed!", text_color="#FF5252")

    def tracking(self):
        self.status_label.configure(text="Tracking started...", text_color="#2196F3")
        print("🧠 Tracking chưa được triển khai...")

    def quit_app(self):
        self.status_label.configure(text="Exiting...", text_color="#FF5252")
        self.cap.release()
        self.root.destroy()

# Chạy ứng dụng
if __name__ == "__main__":
    app = CameraApp(ctk.CTk())
    app.root.mainloop()
