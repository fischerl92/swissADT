import streamlit as st
import os

from cgdetr import CGDETRPredictor
from swiss_adt import save_subclip, extract_frames

if __name__ == "__main__":
    # Set the title of the app
    st.title("SwissADT: Multimodal Audio Description Translation")

    # Get the path of the video file
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    # Get the Audio description
    audio_description = st.text_input("Enter the audio description")

    # Select if user wants number of frames or everny nth frame
    option = st.radio("Select the type of extraction", ("Number of frames", "Every nth frame"))

    if option == "Every nth frame":
        # Get the nth frame to extract
        nth_frame = st.number_input("Enter the nth frame to extract:", min_value=10, max_value=100, value=50)
        num_frames = None
    else:
        nth_frame = None
        num_frames = st.number_input("Enter the number of frames to extract:", min_value=1, max_value=20, value=4)


    # Get Moment
    if st.button("Get Moment"):
        if not video_file or not audio_description:
            st.error("Please upload a video file and enter the audio description.")
            st.stop()

        if not nth_frame and not num_frames:
            st.error("Please select the type of extraction.")
            st.stop()
        
        os.makedirs("tmp", exist_ok=True)

        vid_file = "tmp/input.mp4"
        moment_file = "tmp/moment.mp4"

        with open(vid_file, "wb") as f:
            f.write(video_file.getbuffer())

        # Find the moment
        model = CGDETRPredictor(device="cpu")
        predictions = model.localize_moment(video_path=vid_file, query_list=[audio_description])
        moment = predictions[0]["pred_relevant_windows"][0]
        save_subclip(vid_file, moment_file, moment[0], moment[1])
        
        # Display the moment
        st.divider()
        st.caption(f"Extracted Moment for Audio Description: {audio_description}")
        st.video(moment_file)

        # Extract the frames
        if nth_frame and not num_frames:
            # Extract the frames
            frames = extract_frames(moment_file, nth_frame=nth_frame, num_frames=None)

        else:
            # Extract the frames
            frames = extract_frames(moment_file, num_frames=num_frames, nth_frame=None)

        # Display the frames
        frames = list(frames)
        len_frames = len(frames)

        st.divider()
        st.caption(f"Sending the following frames to the model for translation.")

        if len_frames < 6:
            st.image(frames, width=200)
        else:
            for i in range(0, len_frames, 6):
                st.image(frames[i:i + 6], width=200)

        