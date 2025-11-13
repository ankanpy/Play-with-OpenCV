import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False


class HandMouseController(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe Hands with specific parameters
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.drawing_utils = mp.solutions.drawing_utils

        # Get the size of the screen
        self.screen_width, self.screen_height = pyautogui.size()

        # Variables to keep track of previous hand position
        self.prev_y = None

        # Feature toggles
        self.enable_mouse_control = True
        self.enable_scrolling = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(img, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                self.drawing_utils.draw_landmarks(frame, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS)
                landmarks = hand_landmark.landmark

                # Extract finger positions
                finger_tips = [8, 12, 16, 20]
                finger_mcp = [5, 9, 13, 17]
                finger_states = []

                for tip, mcp in zip(finger_tips, finger_mcp):
                    # Tip above MCP joint means finger is extended
                    if landmarks[tip].y < landmarks[mcp].y:
                        finger_states.append(1)
                    else:
                        finger_states.append(0)

                # Check for Victory sign (Index and middle fingers extended)
                if finger_states == [1, 1, 0, 0]:
                    gesture = "Victory"
                # Check for Spider-Man sign (Index and little fingers extended)
                elif finger_states == [1, 0, 0, 1]:
                    gesture = "Spider-Man"
                else:
                    gesture = "None"

                # Get index finger coordinates
                index_finger_tip = landmarks[8]
                x = int(index_finger_tip.x * frame_width)
                y = int(index_finger_tip.y * frame_height)
                index_x = self.screen_width / frame_width * x
                index_y = self.screen_height / frame_height * y

                # Mouse Control
                if self.enable_mouse_control:
                    # Check for click gesture (thumb and index finger close together)
                    thumb_tip = landmarks[4]
                    thumb_x = int(thumb_tip.x * frame_width)
                    thumb_y = int(thumb_tip.y * frame_height)
                    thumb_index_distance = np.hypot(x - thumb_x, y - thumb_y)

                    if thumb_index_distance < 40:
                        # Click action
                        pyautogui.click()
                        pyautogui.sleep(1)
                    elif thumb_index_distance < 100:
                        # print("distance:", thumb_index_distance)
                        # Move cursor
                        pyautogui.moveTo(index_x, index_y)

                # Scrolling Control
                if self.enable_scrolling and gesture in ["Victory", "Spider-Man"]:
                    # Get current y position
                    current_y = landmarks[0].y  # Use wrist position

                    if self.prev_y is not None:
                        delta_y = self.prev_y - current_y
                        scroll_amount = delta_y * 1000  # Adjust scroll sensitivity

                        if abs(scroll_amount) > 5:
                            pyautogui.scroll(int(scroll_amount))

                    self.prev_y = current_y
                else:
                    self.prev_y = None

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


def main():
    st.set_page_config(page_title="Virtual Mouse Controller", layout="wide")
    st.title("Virtual Mouse Controller")

    st.write(
        """
    Control your computer using hand gestures detected by your webcam.

    ### Instructions:

    - **Move Cursor** (:pinching_hand:): Hold your index finger up and move your hand to move the cursor.
    - **Click** (:ok_hand:): Bring your thumb and index finger close together to click.
    - **Scroll**:
        - **Victory Sign** (:v:): Extend your index and middle fingers to enable scrolling.
        - **Spider-Man Sign** (:the_horns:): Extend your index and little fingers to enable scrolling.
        - Move your hand **up** or **down** to scroll.
    - **PS**: 
        - Ensure good lighting and keep your hand within the webcam's view.
        - Currently PyAutoGUI doesn't support remote/headless machines. Clone the project and run it in your local machine.
    """
    )

    st.sidebar.title("Settings")
    enable_mouse = st.sidebar.checkbox("Enable Mouse Control", value=True)
    enable_scroll = st.sidebar.checkbox("Enable Scrolling", value=True)

    # Start the webcam stream with the HandMouseController
    webrtc_ctx = webrtc_streamer(
        key="hand-mouse",
        video_processor_factory=HandMouseController,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False,
        },
        async_processing=True,
        video_html_attrs={
            "style": {"width": "100%", "height": "auto"},
            "controls": False,
            "autoPlay": True,
        },
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.enable_mouse_control = enable_mouse
        webrtc_ctx.video_processor.enable_scrolling = enable_scroll

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    Developed with ❤️  by **Ankan**

    **Tools:**

    - OpenCV
    - PyAutoGUI
    - MediaPipe
    - Streamlit
    """
    )


if __name__ == "__main__":
    main()