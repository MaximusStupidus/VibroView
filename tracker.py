import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class VideoTracker:
    def __init__(self, video_path, fps, track_axis='magnitude'):
        self.video_path = video_path
        self.fps = fps
        self.track_axis = track_axis
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_number = 0
        self.displacement_list = []
        self.time_list = []
        self.bbox = None
        self.tracker = None
        self.x0, self.y0 = None, None

    def load_video(self):
        if not self.cap.isOpened():
            raise Exception(f"Error: Cannot open video file {self.video_path}")

        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Failed to read the first frame of the video.")
        return frame

    def select_roi(self, frame):
        self.bbox = cv2.selectROI("Tracking", frame, False)
        cv2.destroyWindow("Tracking")
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, self.bbox)
        self.x0 = int(self.bbox[0] + self.bbox[2] / 2)
        self.y0 = int(self.bbox[1] + self.bbox[3] / 2)

    def track_object(self):
        print("Tracking started. Press 'ESC' to stop early.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached.")
                break

            self.frame_number += 1
            success, bbox = self.tracker.update(frame)

            if success:
                x = int(bbox[0] + bbox[2] / 2)
                y = int(bbox[1] + bbox[3] / 2)

                displacement = self.calculate_displacement(x, y)
                current_time = self.frame_number / self.fps

                self.displacement_list.append(displacement)
                self.time_list.append(current_time)

                self.draw_tracking_box(frame, bbox)

            cv2.imshow("Tracking", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("Tracking stopped by user.")
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def calculate_displacement(self, x, y):
        if self.track_axis == 'x':
            return x - self.x0
        elif self.track_axis == 'y':
            return y - self.y0
        elif self.track_axis == 'magnitude':
            return np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
        else:
            raise ValueError("Invalid track_axis value. Choose from 'x', 'y', or 'magnitude'.")

    def draw_tracking_box(self, frame, bbox):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.putText(frame, "Tracking", (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    def get_displacement_and_time(self):
        return np.array(self.displacement_list), np.array(self.time_list)


class DisplacementPlotter:
    def __init__(self, displacement, time, fps, root):
        self.displacement = displacement
        self.time = time
        self.fps = fps
        self.fft_freq = None
        self.fft_magnitude = None
        self.root = root

    def plot_displacement_and_fft(self, canvas, figure):
        figure.clear()

        # Displacement vs Time Plot
        ax1 = figure.add_subplot(121)
        ax1.plot(self.time, self.displacement, label='Displacement')
        ax1.set_title('Displacement vs Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Displacement (pixels)')
        ax1.legend()
        ax1.grid(True)

        # FFT Plot
        N = len(self.displacement)
        fft_values = np.fft.fft(self.displacement)
        self.fft_freq = np.fft.fftfreq(N, d=1 / self.fps)

        idx = np.where(self.fft_freq >= 0)
        self.fft_freq = self.fft_freq[idx]
        fft_values = fft_values[idx]

        self.fft_magnitude = np.abs(fft_values) * 2 / N

        ax2 = figure.add_subplot(122)
        ax2.plot(self.fft_freq, self.fft_magnitude, label='FFT Magnitude')

        # Find and highlight peaks
        peaks, _ = find_peaks(self.fft_magnitude, height=0.1 * np.max(self.fft_magnitude))
        ax2.plot(self.fft_freq[peaks], self.fft_magnitude[peaks], "x", label='Peaks')

        ax2.set_title('Frequency Spectrum of Displacement')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlim(0, self.fps / 2)
        ax2.legend()
        ax2.grid(True)

        # Show top peaks in Tkinter interface
        top_peaks = self.fft_freq[peaks][:5]  # Get the top 5 peak frequencies
        message = f"Detected RPM: {', '.join([f'{60*freq:.2f}' for freq in top_peaks])}"
        tk.Label(self.root, text=message).pack()

        canvas.draw()


class AppController:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Displacement Tracker")

        self.fps = tk.IntVar(value=240)
        self.track_axis = tk.StringVar(value='magnitude')
        self.video_path = tk.StringVar()

        # Video input
        tk.Label(root, text="Select Video").pack()
        tk.Button(root, text="Browse", command=self.browse_file).pack()
        self.video_path_label = tk.Label(root, text="No file selected")
        self.video_path_label.pack()

        # FPS input
        tk.Label(root, text="FPS").pack()
        tk.Entry(root, textvariable=self.fps).pack()

        # Axis selection
        tk.Label(root, text="Track Axis").pack()
        tk.OptionMenu(root, self.track_axis, 'x', 'y', 'magnitude').pack()

        # Plot area
        self.figure = Figure(figsize=(10, 5), dpi=100)  # Adjusted size for side-by-side plots
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

        # Start button
        tk.Button(root, text="Start", command=self.start_tracking).pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.video_path.set(file_path)
        self.video_path_label.config(text=file_path)

    def start_tracking(self):
        if not self.video_path.get():
            tk.messagebox.showwarning("Input Error", "Please select a video file.")
            return

        if self.fps.get() <= 0:
            tk.messagebox.showwarning("Input Error", "Please enter a valid FPS.")
            return

        tracker = VideoTracker(self.video_path.get(), self.fps.get(), self.track_axis.get())
        frame = tracker.load_video()
        tracker.select_roi(frame)
        tracker.track_object()

        displacement, time = tracker.get_displacement_and_time()

        plotter = DisplacementPlotter(displacement, time, self.fps.get(), self.root)
        plotter.plot_displacement_and_fft(self.canvas, self.figure)


# Running the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AppController(root)
    root.mainloop()
