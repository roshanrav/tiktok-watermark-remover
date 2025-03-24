import os
import cv2
import uuid
import subprocess
import numpy as np
import json
from flask import Flask, request, jsonify, send_from_directory, abort
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'processed')
DEBUG_FOLDER = os.path.join(os.getcwd(), 'debug')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

def get_padded_roi(x, y, w, h, frame_width, frame_height, pad_x=5, pad_y=5, extra_width=25, extra_height=50):
    """
    compute a padded ROI
    """
    x_pad = max(0, x - pad_x)
    y_pad = max(0, y - pad_y)
    w_pad = min(w + extra_width, frame_width - x_pad)
    h_pad = min(h + extra_height, frame_height - y_pad)
    return x_pad, y_pad, w_pad, h_pad

def detect_tiktok_watermarks(video_path, debug_id):
    """
    Detect TikTok watermarks throughout the video
    """
    logger.info(f"Detecting TikTok watermarks in {video_path}")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video {video_path}")
        return []
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video info: {width}x{height}, {fps} fps, {duration:.2f} seconds")
    
    if duration <= 15:
        sample_interval = 1  # every 1 second for short videos
    elif duration <= 60:
        sample_interval = 2  # every 2 seconds for medium videos
    else:
        sample_interval = 5  # Every 5 seconds for long videos
    
    # Define color ranges for TikTok watermark (in HSV)
    lower_cyan = np.array([85, 100, 100])
    upper_cyan = np.array([95, 255, 255])
    lower_pink = np.array([150, 100, 100])
    upper_pink = np.array([170, 255, 255])
    
    # Store watermark positions and timestamps
    watermark_regions = []
    
    # Sample frames throughout the video
    for time_sec in range(0, int(duration), sample_interval):
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame at {time_sec}s")
            continue
            
        # Save raw frame for debugging
        debug_frame_path = os.path.join(DEBUG_FOLDER, f"{debug_id}_frame_{time_sec}s.jpg")
        cv2.imwrite(debug_frame_path, frame)
        
        # Convert frame to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for cyan and pink
        cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Save color masks
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{debug_id}_cyan_{time_sec}s.jpg"), cyan_mask)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{debug_id}_pink_{time_sec}s.jpg"), pink_mask)
        
        combined_mask = cv2.bitwise_or(cyan_mask, pink_mask)
        
        # apply morphological operations to connect nearby components
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{debug_id}_mask_{time_sec}s.jpg"), combined_mask)
        
        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # draw contours on a copy of the frame for debugging
        contour_frame = frame.copy()
        cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{debug_id}_contours_{time_sec}s.jpg"), contour_frame)
        
        # find potential watermark regions
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area - TikTok watermarks are usually small but not tiny
            if area > 100 and area < width * height * 0.1: 
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if there are both cyan and pink pixels in this region
                roi_cyan = cv2.countNonZero(cyan_mask[y:y+h, x:x+w])
                roi_pink = cv2.countNonZero(pink_mask[y:y+h, x:x+w])
                
                if roi_cyan > 10 and roi_pink > 10:
                    logger.info(f"Watermark detected at {time_sec}s: ({x},{y}) size {w}x{h}")
                    
                    x_pad, y_pad, w_pad, h_pad = get_padded_roi(x, y, w, h, width, height, pad_x=5, pad_y=5, extra_width=25, extra_height=50)
                    
                    # Save the padded region with its timestamp
                    watermark_regions.append({
                        "timestamp": time_sec,
                        "x": x_pad,
                        "y": y_pad,
                        "width": w_pad,
                        "height": h_pad
                    })
                    
                    cv2.rectangle(contour_frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 0, 255), 2)
                    
        # Save the annotated frame if we found watermarks
        if len(contours) > 0:
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{debug_id}_annotated_{time_sec}s.jpg"), contour_frame)
    
    cap.release()
    
    if watermark_regions:
        logger.info(f"Detected {len(watermark_regions)} watermark positions")
    else:
        logger.info("No watermarks detected")
        
    return watermark_regions


def create_watermark_removal_script(watermark_regions, video_width, video_height, video_fps, debug_id):
    """
    Create an FFmpeg complex filter script to blur watermarks throughout the video.
    Uses the 'boxblur' filter with enable/disable expressions based on timestamps.
    """
    if not watermark_regions:
        logger.warning("No watermark regions to blur")
        return None
    
    # Create a temporary file for the filter script
    script_path = os.path.join(DEBUG_FOLDER, f"{debug_id}_filter_script.txt")
    
    # Group regions by similar positions (to reduce number of filters)
    grouped_regions = []
    for region in watermark_regions:
        # Check if we can merge with an existing group
        merged = False
        for group in grouped_regions:
            # Calculate overlap or proximity
            base = group[0]
            overlap_x = (region['x'] < base['x'] + base['width']) and (base['x'] < region['x'] + region['width'])
            overlap_y = (region['y'] < base['y'] + base['height']) and (base['y'] < region['y'] + region['height'])
            
            if overlap_x and overlap_y:
                # merge by using the union of both regions
                x1 = min(base['x'], region['x'])
                y1 = min(base['y'], region['y'])
                x2 = max(base['x'] + base['width'], region['x'] + region['width'])
                y2 = max(base['y'] + base['height'], region['y'] + region['height'])
                
                base['x'] = x1
                base['y'] = y1
                base['width'] = x2 - x1
                base['height'] = y2 - y1
                
                group.append(region)
                merged = True
                break
                
        if not merged:
            grouped_regions.append([region])
    
    logger.info(f"Grouped {len(watermark_regions)} regions into {len(grouped_regions)} blur zones")
    
    filters = []
    
    filters.append("[0:v]")
    
    # Add split filter to create copies for each blur region
    split_count = len(grouped_regions)
    if split_count > 0:
        filters.append(f"split={split_count}")
        for i in range(split_count):
            filters.append(f"[v{i}]")
        
        # Process each group of regions
        for i, group in enumerate(grouped_regions):
            # Reference the split output
            filters.append(f"[v{i}]")
            
            # For each region,enable blur shortly before and after its timestamp
            enable_exprs = []
            for region in group:
                ts = region['timestamp']
                # Enable blur 0.5s before and 1.5s after the detected timestamp
                enable_exprs.append(f"between(t,{max(0, ts-0.5)},{ts+1.5})")
            
            # Combine enable expressions with OR
            enable_expr = "+".join(enable_exprs)
            if len(enable_exprs) > 1:
                enable_expr = f"({enable_expr}>0)"
            
            # Get coordinates from the base region in this group
            base = group[0]
            x, y, w, h = base['x'], base['y'], base['width'], base['height']
            
            # Apply boxblur with enable/disable expression
            blur_amount = 20  # Adjust blur strength as needed
            filters.append(f"boxblur={blur_amount}:enable='{enable_expr}'[b{i}]")
            
        # Overlay all the blurred regions back onto the original
        for i in range(split_count):
            if i == 0:
                filters.append(f"[0:v][b{i}]overlay=enable='1'")
            else:
                filters.append(f"[v_out{i-1}][b{i}]overlay=enable='1'")
            
            if i < split_count - 1:
                filters.append(f"[v_out{i}]")
            else:
                filters.append(f"[v_out]")
    else:
        # No watermarks to remove, just pass through
        filters.append("null[v_out]")
    
    # Write the filter script
    with open(script_path, 'w') as f:
        f.write(''.join(filters))
    
    logger.info(f"Created FFmpeg filter script: {script_path}")
    return script_path

def remove_moving_watermarks(input_path, output_path, watermark_regions, debug_id):
    if not watermark_regions:
        logger.warning("No watermarks to remove, copying original video")
        subprocess.run(['cp', input_path, output_path])
        return True

    # Get video dimensions
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Could not open video to get dimensions")
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    logger.info(f"Video dimensions: {width}x{height}")

    # Build the filter expression with boundary validation
    filter_exprs = []
    for region in watermark_regions:
        x = max(0, min(region['x'], width-1))
        y = max(0, min(region['y'], height-1))
        w = min(region['width'], width - x)
        h = min(region['height'], height - y)
        
        # Skip invalid regions (delogo requires minimum dimensions)
        if w < 4 or h < 4:
            logger.warning(f"Skipping too small region: {x},{y},{w},{h}")
            continue
            
        timestamp = region['timestamp']
        start_time = max(0, timestamp - 0.5)
        end_time = timestamp + 1.5

        filter_exprs.append(
            f"delogo=x={x}:y={y}:w={w}:h={h}:enable='between(t,{start_time},{end_time})'"
        )

    # If we have no valid filters, just copy the video
    if not filter_exprs:
        logger.warning("No valid watermark regions, copying original video")
        subprocess.run(['cp', input_path, output_path])
        return True
        
    filter_chain = ",".join(filter_exprs)

    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', filter_chain,
        '-c:a', 'copy',
        output_path
    ]

    logger.info(f"Running combined FFmpeg command:\n{' '.join(command)}")

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        logger.error("FFmpeg error:\n" + result.stderr.decode())
        # Fall back to a basic copy if delogo fails
        logger.warning("Falling back to direct copy due to FFmpeg error")
        subprocess.run(['cp', input_path, output_path])
        return False

    return True


def remove_moving_watermarks_inpaint(input_path, output_path, watermark_regions, debug_id):
    logger.info("Starting inpainting based watermark removal")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error("Unable to open video for inpainting")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {frame_count} frames at {fps} fps, resolution {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = output_path + ".temp.mp4"
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current time in seconds
        current_time = frame_idx / fps

        # Create a mask for inpainting
        mask = np.zeros((height, width), dtype=np.uint8)

        # For each detected watermark region, add it to the mask if the current frame time is within its window
        for region in watermark_regions:
            ts = region['timestamp']
            start_time = max(0, ts - 0.5)
            end_time = ts + 1.5
            if start_time <= current_time <= end_time:
                x = int(region['x'])
                y = int(region['y'])
                w = int(region['width'])
                h = int(region['height'])
                # Draw a filled white rectangle on the mask for this region
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # If the mask has any white areas, apply inpainting
        if cv2.countNonZero(mask) > 0:
            inpaint_radius = 3  
            frame = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            logger.info(f"Inpainting progress: {frame_idx}/{frame_count} frames processed")

    cap.release()
    writer.release()
    logger.info(f"Inpainting processed {frame_idx} frames.")

    # Merge the original audio into the processed video using FFmpeg
    final_output = output_path
    command = [
        'ffmpeg', '-y',
        '-i', temp_video_path,
        '-i', input_path,
        '-c:v', 'copy',
        '-c:a', 'aac',        
        '-map', '0:v:0',      
        '-map', '1:a?',      
        '-shortest',
        final_output
    ]
    logger.info("Merging audio with command: " + " ".join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error("Audio merge error:\n" + result.stderr.decode())
        return False

    os.remove(temp_video_path)
    return True

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        debug_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting processing request {debug_id}")
        
        filename = f"{debug_id}.mp4"
        input_video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_video_path)
        
        filesize = os.path.getsize(input_video_path)
        logger.info(f"Saved file: {input_video_path}, size: {filesize} bytes")
        
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {input_video_path}")
            return jsonify({'error': 'Invalid video file'}), 400
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        logger.info(f"Video info: {width}x{height}, {fps} fps, {frame_count} frames")
        
        logger.info("Starting watermark detection...")
        watermark_regions = detect_tiktok_watermarks(input_video_path, debug_id)
        
        watermark_found = len(watermark_regions) > 0
        logger.info(f"Watermark detection result: {watermark_found}")
        
        with open(os.path.join(DEBUG_FOLDER, f"{debug_id}_watermarks.json"), 'w') as f:
            json.dump(watermark_regions, f, indent=2)
        
        output_filename = f"processed_{filename}"
        output_video_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Read the chosen method (default to "blur")
        method = request.form.get('method', 'blur')
        
        if method == 'inpaint':
            logger.info("Using inpainting method for watermark removal")
            success = remove_moving_watermarks_inpaint(input_video_path, output_video_path, watermark_regions, debug_id)
        else:
            logger.info("Using blur method for watermark removal")
            success = remove_moving_watermarks(input_video_path, output_video_path, watermark_regions, debug_id)
        
        if not success:
            logger.warning("Watermark removal had issues, but a video was still produced")
        
        if not os.path.exists(output_video_path):
            logger.error("Output video file doesn't exist after processing")
            subprocess.run(['cp', input_video_path, output_video_path])
        
        processed_video_url = request.host_url + 'processed/' + output_filename
        debug_frames_url = request.host_url + 'debug'
        
        return jsonify({
            'watermark_detected': watermark_found,
            'watermark_count': len(watermark_regions),
            'processed_video_url': processed_video_url,
            'debug_info': {
                'id': debug_id,
                'frames_url': debug_frames_url,
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count
                }
            }
        })
    except Exception as e:
        import traceback
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/processed/<filename>')
def serve_processed_video(filename):
    try:
        return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

@app.route('/debug/<filename>')
def serve_debug_file(filename):
    try:
        return send_from_directory(DEBUG_FOLDER, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

@app.route('/debug')
def list_debug_files():
    files = [f for f in os.listdir(DEBUG_FOLDER) if os.path.isfile(os.path.join(DEBUG_FOLDER, f))]
    return jsonify({
        'debug_files': files,
        'count': len(files)
    })

if __name__ == '__main__':
    logger.info("Starting TikTok watermark detection and removal server...")
    app.run(host='0.0.0.0', port=5001, debug=True)
