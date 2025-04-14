# Python In-built packages
from pathlib import Path
import PIL
import supervision as sv
from pydub.playback import play
from playsound import playsound
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from collections import defaultdict
import cv2
from PIL import Image
import numpy as np

def play_sound():
    """Play system beep sound"""
    try:
        import winsound
        winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms
    except Exception as e:
        st.error(f"Error playing sound: {str(e)}")

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'video_processed_frames' not in st.session_state:
    st.session_state.video_processed_frames = 0
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0

my_email = "vivekraina33.vr@gmail.com"
password_key = "xsfuoajtzwfqhnvl"
gmail_server = "smtp.gmail.com"
gmail_port = 587

# Email setup code...
my_server = smtplib.SMTP(gmail_server, gmail_port)
my_server.ehlo()
my_server.starttls()
my_server.login(my_email, password_key)
message = MIMEMultipart("alternative")
text_content = "Alert: Crowd/Violence Detection Alert"
message.attach(MIMEText(text_content))
recruiter_email = "ayushtiwari.creatorslab@gmail.com"
msg_string = message.as_string()

import settings
import helper

class CrowdAnalytics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.people_counts = []
        self.violence_detections = []
        self.alert_triggered = False
        self.max_people_count = 0
        self.highest_density = "Very Low"
    
    def update(self, count=0, detections=None):
        if count > 0:
            self.people_counts.append(count)
            self.max_people_count = max(self.max_people_count, count)
            self._update_density(count)
            
        if detections:
            if isinstance(detections, list):
                self.violence_detections.extend([d for d in detections if d in ["Violence", "NonViolence"]])
    
    def _update_density(self, count):
        if count >= 50:
            self.highest_density = "High"
        elif count >= 10:
            if self.highest_density not in ["High"]:
                self.highest_density = "Medium"
        elif count >= 1:
            if self.highest_density not in ["High", "Medium"]:
                self.highest_density = "Low"
    
    def get_final_stats(self):
        stats = {
            'max_people': self.max_people_count,
            'density': self.highest_density,
            'violence_detected': "Violence" in self.violence_detections
        }
        return stats

def process_frame(frame, model, model_name, confidence, analytics):
    """Process a single frame and update analytics"""
    try:
        if model_name == "violenceguns":
            # Convert frame to RGB if it's BGR
            if isinstance(frame, np.ndarray) and frame.shape[-1] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Convert to PIL Image if needed
            if isinstance(frame_rgb, np.ndarray):
                frame_rgb = Image.fromarray(frame_rgb)

            res = model.infer(frame_rgb, confidence=confidence)
            detections = sv.Detections.from_inference(res[0].dict(by_alias=True, exclude_none=True))
            
            # Safely handle class names
            class_names = []
            if 'class_name' in detections.data:
                if isinstance(detections.data['class_name'], np.ndarray):
                    class_names = detections.data['class_name'].tolist()
                elif isinstance(detections.data['class_name'], list):
                    class_names = detections.data['class_name']
                else:
                    class_names = [detections.data['class_name']]
            
            analytics.update(detections=class_names)
            
            # Convert back to numpy array for annotation if needed
            frame_for_annotation = np.array(frame_rgb) if isinstance(frame_rgb, Image.Image) else frame_rgb
            
            # Annotate frame
            bounding_box_annotator = sv.RoundBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            frame = bounding_box_annotator.annotate(scene=frame_for_annotation, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)
            
        elif model_name == "crowd":
            res = model.predict(frame, conf=confidence)
            names = res[0].names
            class_detections = {}
            
            for k, v in names.items():
                count = res[0].boxes.cls.tolist().count(k)
                class_detections[v] = count
                
            if 'people' in class_detections:
                analytics.update(count=class_detections['people'])
            
            frame = res[0].plot()[:, :, ::-1]
        
        return frame
        
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return frame

def process_video(video_path, model, model_name, confidence, analytics, progress_bar, status_text):
    """Process video and return final statistics"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.total_frames = total_frames
        
        if total_frames <= 0:
            st.error("Could not read video frames")
            return None
            
        frame_placeholder = st.empty()
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frames += 1
            progress = int((processed_frames / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {processed_frames}/{total_frames}")
            
            # Process frame
            processed_frame = process_frame(frame, model, model_name, confidence, analytics)
            
            # Display frame
            if processed_frame is not None:
                if model_name == "violenceguns":
                    frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                else:
                    frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            # Update session state
            st.session_state.video_processed_frames = processed_frames
            
        cap.release()
        return analytics.get_final_stats()
        
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return None

# Main Streamlit UI
st.set_page_config(
    page_title="Crowd and Violence Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Crowd and Violence Detection")

# Initialize analytics
crowd_analytics = CrowdAnalytics()

# Sidebar configuration
st.sidebar.header("Model Config")

model_type = st.sidebar.radio(
    "Select Task", ['Crowd Detection', 'Violence Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 1, 100, 40)) / 100

# Model selection
if model_type == 'Crowd Detection':
    model_path = Path(settings.DETECTION_MODEL1)
    model_name = "crowd"
elif model_type == 'Violence Detection':
    model_path = Path(settings.DETECTION_MODEL2)
    model_name = "violenceguns"

# Load model
try:
    if model_name == "crowd":
        model = helper.load_model(model_path)
    elif model_name == "violenceguns":
        model = helper.load_model_vguns(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Source selection
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Source", settings.SOURCES_LIST)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect'):
                crowd_analytics.reset()
                
                if model_name == "violenceguns":
                    res = model.infer(uploaded_image,
                                    confidence=confidence)
                    detections = sv.Detections.from_inference(res[0].dict(by_alias=True, exclude_none=True))
                    bounding_box_annotator = sv.RoundBoxAnnotator()
                    label_annotator = sv.LabelAnnotator()

                    annotated_image = bounding_box_annotator.annotate(
                        scene=uploaded_image, detections=detections)
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=detections)
                    st.image(annotated_image, caption='Detected Image',
                            use_column_width=True)
                            
                    class_names = detections.data['class_name']
                    threat_detected = False
                    
                    for class_name in class_names:
                        if class_name == "Violence":
                            threat_detected = True
                            st.error("Violence Detected!")
                            play_sound()
                            st.warning("Alert Sound Played 🚨")
                            my_server.sendmail(from_addr=my_email, 
                                            to_addrs=recruiter_email, 
                                            msg=msg_string)
                            st.success("Alert Email Sent")
                    
                    if not threat_detected:
                        st.success("No Violence Detected")
                        
                elif model_name == "crowd":
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    names = res[0].names
                    class_detections_values = []
                    for k, v in names.items():
                        class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                    classes_detected = dict(zip(names.values(), class_detections_values))
                    
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                    
                    if 'people' in classes_detected:
                        no_of_people = classes_detected['people']
                        if no_of_people:
                            if no_of_people == 0:
                                st.info("Crowd Density: Very Low")
                                st.success("No of people: {}".format(no_of_people))
                            elif 1< no_of_people < 10:
                                st.info("Crowd Density: Low")
                                st.success("No of people: {}".format(no_of_people))
                            elif 10< no_of_people <50:
                                st.info("Crowd Density: Medium")
                                st.success("No of people: {}".format(no_of_people))
                                play_sound()
                                st.warning("Alert Sound Played 🚨")
                            if no_of_people > 50:
                                st.title("Crowd Density: High")
                                st.title("No of people: {}".format(no_of_people))
                                play_sound()
                                my_server.sendmail(from_addr=my_email, 
                                                to_addrs=recruiter_email, 
                                                msg=msg_string)
                                st.success("Alert Email Sent")
                        else:
                            st.title("No People")

elif source_radio == settings.VIDEO:
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        if st.sidebar.button('Process Video'):
            # Reset analytics
            crowd_analytics.reset()
            st.session_state.processing_complete = False
            
            # Create progress elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded video temporarily
                video_path = f"temp_video_{int(time.time())}.mp4"
                with open(video_path, 'wb') as f:
                    f.write(uploaded_video.read())
                
                # Process video
                final_stats = process_video(
                    video_path, model, model_name, confidence, 
                    crowd_analytics, progress_bar, status_text
                )
                
                # Show final results
                st.success("Video Processing Complete!")
                
                with st.container():
                    st.header("Final Analysis Results")
                    
                    if model_name == "crowd":
                        st.info(f"Maximum Crowd Density: {final_stats['density']}")
                        st.success(f"Peak People Count: {final_stats['max_people']}")
                        
                        if final_stats['density'] in ["Medium", "High"]:
                            st.warning("⚠️ High crowd density detected!")
                            play_sound()
                            if final_stats['density'] == "High":
                                my_server.sendmail(from_addr=my_email, 
                                                to_addrs=recruiter_email, 
                                                msg=msg_string)
                                st.success("Alert Email Sent")

                    elif model_name == "violenceguns":
                        if final_stats['violence_detected']:
                            st.error("⚠️ Violence Detected in Video!")
                            play_sound()
                            my_server.sendmail(from_addr=my_email, 
                                            to_addrs=recruiter_email, 
                                            msg=msg_string)
                            st.success("Alert Email Sent")
                        else:
                            st.success("✅ No threats detected in video")
                
                # Cleanup
                import os
                os.remove(video_path)
                
            except Exception as ex:
                st.error(f"Error processing video: {str(ex)}")
            finally:
                progress_bar.empty()
                status_text.empty()
                st.session_state.processing_complete = True

elif source_radio == settings.WEBCAM:
    if st.sidebar.button('Start Webcam'):
        st.info("Webcam processing will start... (Implementation similar to video)")
        
elif source_radio == settings.RTSP:
    rtsp_url = st.sidebar.text_input("RTSP URL")
    if st.sidebar.button('Start RTSP Stream'):
        st.info("RTSP stream processing will start... (Implementation similar to video)")
        
elif source_radio == settings.YOUTUBE:
    youtube_url = st.sidebar.text_input("YouTube URL")
    if st.sidebar.button('Process YouTube Video'):
        st.info("YouTube video processing will start... (Implementation similar to video)")
        
else:
    st.error("Please select a valid source type!")

# Cleanup at the end
if st.session_state.processing_complete:
    my_server.quit()  # Clean server shutdown