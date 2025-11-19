import torch
import torch.nn as nn
import torchvision
import os
import time
import numpy as np
from PIL import Image
import cv2

def is_low_light(frame, threshold=0.20):
    """
    Detect whether the given frame is captured in low light.

    Args:
        frame: BGR image (numpy array)
        threshold: mean Y (luminance) threshold in [0,1].
                   Lower values => darker image. Default 0.35 works for 8-bit images.

    Returns:
        (is_low_light, mean_Y)
        is_low_light: True if mean luminance < threshold
        mean_Y: mean luminance value in [0,1]
    """
    # Convert to YCrCb and extract luminance channel
    y_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean_Y = float(np.mean(y_channel) / 255.0)

    return mean_Y < threshold, mean_Y

USE_FP16 = True
TARGET_WIDTH = 640
class enhance_net_nopool(nn.Module):
    def __init__(self):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1, enhance_image, r


def sample_frames_to_6fps(input_path):
    """
    Sample video frames to 6 FPS without saving to disk

    Args:
        input_path: Path to input video

    Returns:
        frames: List of sampled frames
        width: Video width
        height: Video height
    """
    # Read input
    cap = cv2.VideoCapture(input_path)

    # Get original fps and size
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Sampling video from {orig_fps} FPS to 6 FPS...")

    frame_interval = int(orig_fps / 6)  # sample roughly 6 fps

    frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_interval == 0:
            frames.append(frame)
        frame_id += 1

    cap.release()
    print(f"Sampling complete! Collected {len(frames)} frames from {frame_id} total frames.")
    return frames, width, height


def enhance_video(video_path, model_path, output_path, batch_size=12, skip_frames=0):
    """
    Process a video file through the enhancement network

    Args:
        video_path: Path to input video
        model_path: Path to trained model weights
        output_path: Path to save enhanced video
        batch_size: Number of frames to process at once (default: 1)
        skip_frames: Process every Nth frame (0 = process all frames)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load the model in a device-aware way
    DCE_net = enhance_net_nopool().to(device)
    # map_location ensures loading works whether model was saved on CPU or GPU
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))
    DCE_net.eval()
    # Enable FP16 only when running on CUDA
    if USE_FP16 and device.type == 'cuda':
        DCE_net.half()
    # Open input video to check FPS
    cap = cv2.VideoCapture(video_path)
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    print(f"Original video FPS: {orig_fps}")

    # Check if FPS is greater than 6 and sample frames if needed
    if orig_fps > 6:
        print(f"FPS ({orig_fps}) is greater than 6. Sampling frames to 6 FPS...\n")

        # Sample frames directly without saving
        frames, width, height = sample_frames_to_6fps(video_path)
        total_frames = len(frames)
        fps = 6
        use_sampled_frames = True
    else:
        print(f"FPS is {orig_fps}. Proceeding with enhancement...\n")

        # Open the video normally
        cap = cv2.VideoCapture(video_path)
        fps = orig_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = None
        use_sampled_frames = False



    print(f"Video properties for enhancement:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_time = 0

    print("\nProcessing video...")

    with torch.no_grad():
        if use_sampled_frames:
            # Process pre-sampled frames
            for frame_count, frame in enumerate(frames, 1):
                start_time = time.time()

                # low_light, meanY = is_low_light(frame)

                # if low_light:
                #     print(f"Frame {frame_count}: low-light detected (meanY={meanY:.3f})")
                # else:
                #     print(f"Frame {frame_count}: normal lighting (meanY={meanY:.3f})")

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Normalize and convert to tensor
                data_lowlight = torch.from_numpy(frame_rgb / 255.0).float()
                data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)
                if USE_FP16 and device.type == 'cuda':
                    data_lowlight = data_lowlight.half()

                # Enhance the frame
                _, enhanced_image, _ = DCE_net(data_lowlight)

                # Convert back to numpy array
                enhanced_frame = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)

                # Convert RGB back to BGR for OpenCV
                enhanced_frame_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(enhanced_frame_bgr)

                # Calculate processing time
                frame_time = time.time() - start_time
                total_time += frame_time

                # Display progress
                if frame_count % 30 == 0:
                    avg_fps = frame_count / total_time
                    eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                    print(f"Frame {frame_count}/{total_frames} | "
                          f"Processing: {frame_time*1000:.2f}ms/frame | "
                          f"Speed: {avg_fps:.2f} FPS | "
                          f"ETA: {eta:.1f}s")
        else:
            # Process frames from video file
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # low_light, meanY = is_low_light(frame)

                # if low_light:
                #     print(f"Frame {frame_count}: low-light detected (meanY={meanY:.3f})")
                # else:
                #     print(f"Frame {frame_count}: normal lighting (meanY={meanY:.3f})")

                frame_count += 1
                start_time = time.time()

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Normalize and convert to tensor
                data_lowlight = torch.from_numpy(frame_rgb / 255.0).float()
                data_lowlight = data_lowlight.permute(2, 0, 1).unsqueeze(0).to(device)
                if USE_FP16 and device.type == 'cuda':
                    data_lowlight = data_lowlight.half()

                # Enhance the frame
                _, enhanced_image, _ = DCE_net(data_lowlight)

                # Convert back to numpy array
                enhanced_frame = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                enhanced_frame = np.clip(enhanced_frame * 255, 0, 255).astype(np.uint8)

                # Convert RGB back to BGR for OpenCV
                enhanced_frame_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

                # Write frame
                out.write(enhanced_frame_bgr)

                # Calculate processing time
                frame_time = time.time() - start_time
                total_time += frame_time

                # Display progress
                if frame_count % 30 == 0:
                    avg_fps = frame_count / total_time
                    eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                    print(f"Frame {frame_count}/{total_frames} | "
                          f"Processing: {frame_time*1000:.2f}ms/frame | "
                          f"Speed: {avg_fps:.2f} FPS | "
                          f"ETA: {eta:.1f}s")

    # Release resources
    if not use_sampled_frames:
        cap.release()
    out.release()

    # Calculate final statistics
    final_frame_count = total_frames if use_sampled_frames else frame_count
    avg_time = total_time / final_frame_count if final_frame_count > 0 else 0
    avg_fps = final_frame_count / total_time if total_time > 0 else 0

    print(f"\n✓ Processing complete!")
    print(f"  Total frames processed: {final_frame_count}")
    print(f"  Average processing time: {avg_time*1000:.2f}ms/frame")
    print(f"  Average processing speed: {avg_fps:.2f} FPS")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Enhanced video saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    video_path = r"sample3.mp4"
    model_path = r"Epoch99.pth"
    output_path = "enhanced_video.mp4"

    enhance_video(video_path, model_path, output_path)
