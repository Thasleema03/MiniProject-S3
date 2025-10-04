# ===============================
# app.py - Professional Bone Fracture Detection with Navigation
# ===============================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import time

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bone Fracture Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------
# WORKING Grad-CAM Function
# ---------------------------
def guided_gradcam_streamlit(model, image, class_idx=0, alpha=0.4, threshold=0.6):
    """
    Generate Grad-CAM overlay for Streamlit
    class_idx=0 -> Fractured, class_idx=1 -> Not Fractured
    """
    try:
        orig_size = image.size
        img_resized = image.convert("RGB").resize((224,224))
        img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0).astype(np.float32)

        # Find the last convolutional layer
        last_conv_layer = None
        # Try common MobileNetV2 layer names
        layer_names = [
            "out_relu", 
            "block_16_project_BN", 
            "Conv_1", 
            "block_13_expand_relu",
            "block_16_depthwise_relu",
            "block_16_project"
        ]
        
        for layer_name in layer_names:
            try:
                last_conv_layer = model.get_layer(layer_name)
                break
            except:
                continue
        
        # If no specific layer found, find the last convolutional layer
        if last_conv_layer is None:
            for layer in reversed(model.layers):
                if any(keyword in layer.name.lower() for keyword in ['conv', 'block', 'out']):
                    if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                        last_conv_layer = layer
                        break
        
        if last_conv_layer is None:
            st.warning("Using default layer - Grad-CAM may not be optimal")
            # Use the last compatible layer
            for layer in reversed(model.layers):
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break

        # Create grad model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Handle different prediction output formats
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            # Get the correct class probability
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:
                # Multi-class classification
                loss = predictions[:, class_idx]
            else:
                # Binary classification (sigmoid output)
                loss = predictions[:, 0]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Avoid division by zero
        if grads is None:
            st.warning("Gradients are None - using simple Grad-CAM")
            weights = tf.reduce_mean(conv_outputs, axis=(0, 1, 2))
        else:
            # Guided Grad-CAM
            guided_grads = tf.cast(conv_outputs > 0, tf.float32) * tf.cast(grads > 0, tf.float32) * grads
            weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

        # Compute Grad-CAM heatmap
        cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[0, :, :, i].numpy()

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        # Resize to original image size
        heatmap = cv2.resize(cam, orig_size)
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

        # Threshold weak activations
        if threshold > 0:
            heatmap[heatmap < threshold] = 0

        # Normalize again after thresholding
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max

        # Convert to color heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Convert original image to array
        orig_img_array = np.array(image.convert("RGB"))
        
        # Resize heatmap to match original image if sizes differ
        if heatmap_color.shape[:2] != orig_img_array.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (orig_img_array.shape[1], orig_img_array.shape[0]))

        # Overlay on original image
        overlay = cv2.addWeighted(orig_img_array, 1 - alpha, heatmap_color, alpha, 0)
        return Image.fromarray(overlay)
    
    except Exception as e:
        st.error(f"Grad-CAM error: {str(e)}")
        # Return original image as fallback
        return image

# ---------------------------
# Model Loading
# ---------------------------
MODEL_FILE = "mobilenetv2_model_latest.h5"
model_exists = os.path.exists(MODEL_FILE)

@st.cache_resource
def load_model():
    if model_exists:
        try:
            model = tf.keras.models.load_model(MODEL_FILE)
            return model, True
        except Exception as e:
            return None, False
    else:
        return None, False

model, is_real_model = load_model()

# ---------------------------
# Image Preprocessing
# ---------------------------
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ---------------------------
# Page Functions
# ---------------------------
def render_home_page():
    """Render the home/welcome page"""
    st.title(" 🦴Bone Fracture Detection ")
    
    # Hero Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
                                          
        
        ---------------------------------
        
        -  **Accurate Detection**: Advanced deep learning models for fracture identification
        -  **Single Image Analysis**: Detailed analysis of one X-ray
        -  **Batch Processing**: Analyze multiple X-rays simultaneously
        -  **Explainable AI**: Grad-CAM heatmaps showing model focus areas

        """)
        

    # How to Use Section
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. **Select Analysis Mode**: Choose between single or batch analysis from the sidebar
        2. **Upload Images**: Select your X-ray images (JPG, JPEG, PNG formats)
        3. **Configure Settings**: Adjust Grad-CAM options if needed
        4. **Run Analysis**: Click the analyze button to process images
        5. **Review Results**: Examine probabilities and heatmap visualizations
        6. **Interpret Findings**: Use the heatmaps to understand AI focus areas
        """)

def render_single_analysis():
    """Render single image analysis page"""
    st.header(" Single Image Analysis")
    st.markdown("Upload a single X-ray image for detailed fracture analysis with AI-powered insights.")
    
    uploaded_file = st.file_uploader("Upload X-ray Image", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Upload a clear X-ray image for analysis")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption=" Original X-ray Image", use_container_width=True)
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]}x{image.size[1]} pixels")
        
        with col2:
            analyze_btn = st.button(" Analyze Image", type="primary", use_container_width=True)
            
            if analyze_btn:
                if model is None:
                    st.error(" Model not loaded. Please check model file.")
                else:
                    with st.spinner("Analyzing image..."):
                        # Preprocess and predict
                        img_array = preprocess_image(image)
                        start_time = time.time()
                        predictions = model.predict(img_array, verbose=0)
                        inference_time = time.time() - start_time
                        
                        # Process predictions
                        if isinstance(predictions, list):
                            pred = predictions[0]
                        else:
                            pred = predictions
                        
                        if len(pred.shape) == 2 and pred.shape[1] > 1:
                            prob_fractured = float(pred[0][0])
                            prob_healthy = float(pred[0][1])
                        else:
                            prob_healthy = float(pred[0][0])
                            prob_fractured = 1 - prob_healthy
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        result_col1, result_col2 = st.columns(2)
                        with result_col1:
                            st.metric("Fracture Probability", f"{prob_fractured:.2%}",
                                     delta=f"{(prob_fractured-0.5):.2%}" if prob_fractured > 0.5 else None)
                        with result_col2:
                            st.metric("Not Fracture Probability", f"{prob_healthy:.2%}",
                                     delta=f"{(prob_healthy-0.5):.2%}" if prob_healthy > 0.5 else None)
                        
                        # Diagnosis
                        fracture_detected = prob_fractured > 0.5
                        if fracture_detected:
                            st.error(f" **FRACTURE DETECTED** (Confidence: {prob_fractured:.2%})")
                        else:
                            st.success(f" **NO FRACTURE DETECTED** (Confidence: {prob_healthy:.2%})")
                        
                        st.info(f"⏱️ Inference time: {inference_time:.2f} seconds")
                        
                        # Store results in session state
                        st.session_state.last_analysis = {
                            'image': image,
                            'prob_fractured': prob_fractured,
                            'fracture_detected': fracture_detected,
                            'filename': uploaded_file.name
                        }
                        st.session_state.show_gradcam = True
            
            # Grad-CAM generation
            if (st.session_state.last_analysis and enable_gradcam and is_real_model and 
                st.session_state.show_gradcam):
                analysis_data = st.session_state.last_analysis
                
                # Check if we should show Grad-CAM
                show_cam = True
                if show_gradcam_only_fracture and not analysis_data['fracture_detected']:
                    show_cam = False
                    st.info("ℹ️ Grad-CAM is configured to show only for fracture cases.")
                
                if show_cam:
                    st.subheader(" Grad-CAM Visualization")
                    
                    # Generate Grad-CAM button
                    if st.button(" Generate Grad-CAM Heatmap", use_container_width=True):
                        with st.spinner("Generating Grad-CAM overlay..."):
                            try:
                                gradcam_img = guided_gradcam_streamlit(
                                    model, analysis_data['image'],
                                    class_idx=0,
                                    alpha=heatmap_intensity,
                                    threshold=threshold_value
                                )
                                
                                # Display both images side by side
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(analysis_data['image'], 
                                            caption="Original X-ray", 
                                            use_container_width=True)
                                with col2:
                                    st.image(gradcam_img, 
                                            caption="Grad-CAM Overlay (Guided Method)", 
                                            use_container_width=True)
                                
                                # Explanation
                                with st.expander("ℹ️ Understanding the Heatmap"):
                                    st.markdown(f"""
                                    **Heatmap Interpretation:**
                                    - 🔴 **Red areas**: High model attention (potential fracture regions)
                                    - 🔵 **Blue areas**: Low model attention  
                                    - **Method**: Guided Grad-CAM
                                    - **Intensity**: {heatmap_intensity}
                                    - **Threshold**: {threshold_value}
                                    
                                    **What to look for:**
                                    - The model highlights bone fracture lines and suspicious regions
                                    - Focus on areas where the heatmap aligns with visible bone structures
                                    - Red areas indicate where the model detected potential fractures
                                    """)
                                    
                            except Exception as e:
                                st.error(f"Grad-CAM failed: {str(e)}")

def render_batch_analysis():
    """Render batch analysis page"""
    st.header(" Batch Image Analysis")
    st.markdown("Upload multiple X-ray images for efficient batch processing and comparative analysis.")
    
    uploaded_files = st.file_uploader("Upload multiple X-ray images", 
                                     type=["jpg", "jpeg", "png"], 
                                     accept_multiple_files=True,
                                     help="Select multiple images for batch processing")
    
    if uploaded_files:
        st.success(f" {len(uploaded_files)} images selected for analysis")
        
        # Process batch button
        if st.button(" Process All Images", type="primary", use_container_width=True):
            if model is None:
                st.error(" Model not loaded. Please check model file.")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Process image
                    image = Image.open(uploaded_file)
                    img_array = preprocess_image(image)
                    
                    # Predict
                    predictions = model.predict(img_array, verbose=0)
                    
                    if isinstance(predictions, list):
                        pred = predictions[0]
                    else:
                        pred = predictions
                    
                    if len(pred.shape) == 2 and pred.shape[1] > 1:
                        prob_fractured = float(pred[0][0])
                    else:
                        prob_healthy = float(pred[0][0])
                        prob_fractured = 1 - prob_healthy
                    
                    status = "Fracture" if prob_fractured > 0.5 else "Normal"
                    
                    results.append({
                        "Filename": uploaded_file.name,
                        "Fracture Probability": f"{prob_fractured:.2%}",
                        "Status": status,
                        "Image": image,
                        "Prob_Fractured": prob_fractured,
                    })
                
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.batch_results = results
                st.session_state.show_batch_gradcam = True
                
                # Display summary table
                st.subheader("📋 Batch Results Summary")
                summary_data = [{"Filename": r["Filename"], 
                               "Fracture Probability": r["Fracture Probability"],
                               "Status": r["Status"]} for r in results]
                st.table(summary_data)
        
        # Batch Grad-CAM section
        if (st.session_state.batch_results and enable_gradcam and is_real_model and 
            st.session_state.get('show_batch_gradcam', False)):
            
            st.subheader(" Batch Grad-CAM Visualization")
            
            # Filter images for Grad-CAM
            images_to_process = st.session_state.batch_results
            if show_gradcam_only_fracture:
                images_to_process = [r for r in st.session_state.batch_results if r["Status"] == "Fracture"]
                st.info(f"📊 Found {len(images_to_process)} fracture cases for Grad-CAM visualization")
            
            if images_to_process:
                if st.button(" Generate Grad-CAM for All Selected Cases", use_container_width=True):
                    for i, result in enumerate(images_to_process):
                        with st.expander(f" {result['Filename']} - {result['Status']} ({result['Fracture Probability']})", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(result["Image"], 
                                        caption="Original X-ray", 
                                        use_container_width=True)
                            
                            with col2:
                                with st.spinner(f"Generating heatmap {i+1}/{len(images_to_process)}..."):
                                    try:
                                        gradcam_img = guided_gradcam_streamlit(
                                            model, result["Image"],
                                            class_idx=0,
                                            alpha=heatmap_intensity,
                                            threshold=threshold_value
                                        )
                                        st.image(gradcam_img, 
                                                caption="Grad-CAM (Guided)", 
                                                use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Grad-CAM failed: {str(e)}")
                            
                            # Quick stats
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.metric("Fracture Probability", result['Fracture Probability'])
                            with col_stat2:
                                st.metric("Status", result['Status'])

# ---------------------------
# Main Application
# ---------------------------

# Initialize session state
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'show_gradcam' not in st.session_state:
    st.session_state.show_gradcam = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# ---------------------------
# Sidebar Navigation
# ---------------------------
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Home", "Single Image Analysis", "Batch Analysis"],
    index=0,
    label_visibility="collapsed"
)

# Update current page based on selection
if app_mode != st.session_state.current_page:
    st.session_state.current_page = app_mode
    # Reset some states when changing pages
    if app_mode == "Home":
        st.session_state.show_gradcam = False
        st.session_state.show_batch_gradcam = False

# Grad-CAM options - only show when model is loaded
if is_real_model and app_mode != "Home":
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Grad-CAM Options")
    enable_gradcam = st.sidebar.checkbox("Enable Grad-CAM", True, 
                                        help="Show heatmap overlay for fracture areas")
    
    if enable_gradcam:
        heatmap_intensity = st.sidebar.slider("Grad-CAM Intensity", 0.3, 0.8, 0.5, 0.1)
        threshold_value = st.sidebar.slider("Grad-CAM Threshold", 0.2, 0.7, 0.4)
        show_gradcam_only_fracture = st.sidebar.checkbox("Show only for fracture cases", True,
                                                        help="Display Grad-CAM only when fracture is detected")
else:
    enable_gradcam = False


# ---------------------------
# Page Routing
# ---------------------------
if st.session_state.current_page == "Home":
    render_home_page()
elif st.session_state.current_page == "Single Image Analysis":
    render_single_analysis()
elif st.session_state.current_page == "Batch Analysis":
    render_batch_analysis()
