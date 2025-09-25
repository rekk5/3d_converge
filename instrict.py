import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def calibrate_camera(images_folder, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate camera from checkerboard images
    
    Args:
        images_folder: Folder containing the calibration images
        checkerboard_size: Size of the checkerboard (inner corners)
        square_size: Size of squares in meters (for real-world scale)
    
    Returns:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Scale to real-world units
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(f'{images_folder}/*.jpg') + glob.glob(f'{images_folder}/*.png')
    
    if len(images) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    print(f"Found {len(images)} images for calibration")
    
    # Process each image
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Optional: Draw and display the corners
            cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
            cv2.imshow('Corners', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    if len(objpoints) == 0:
        raise ValueError("No chessboard corners detected in any images")
    
    print(f"Successfully detected corners in {len(objpoints)} images")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculate reprojection error to check calibration quality
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Average reprojection error: {mean_error/len(objpoints)}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

def estimate_depth_scale(image_rgb, image_depth, camera_matrix, 
                          object_real_size=0.1, visualize=True):
    """
    Estimate depth scale from a known object size
    
    Args:
        image_rgb: RGB image containing a square object of known size
        image_depth: Corresponding depth image
        camera_matrix: Camera intrinsic matrix
        object_real_size: Real-world size of the object in meters
        visualize: Whether to visualize the detected object
    
    Returns:
        depth_scale: Scale factor to convert depth values to meters
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    
    # Simple square detection (can be replaced with more robust methods)
    # For the sake of example, let's use a simple threshold and contour approach
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    if visualize:
        img_copy = image_rgb.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title("Detected Object")
        plt.show()
    
    # Get depth values for the object
    object_depth = image_depth[y:y+h, x:x+w]
    avg_depth_value = np.median(object_depth)  # Using median to be robust to outliers
    
    # Calculate apparent size in pixels
    apparent_size_pixels = max(w, h)
    
    # Calculate focal length (average of fx and fy)
    focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
    
    # Calculate actual object size in world units based on the perspective projection formula
    # object_size_meters = (object_size_pixels * depth) / focal_length
    # Rearranging to get depth_scale:
    depth_scale = (object_real_size * focal_length) / (apparent_size_pixels * avg_depth_value)
    
    return depth_scale

if __name__ == "__main__":
    # Example usage
    # 1. Camera calibration
    calibration_folder = "calibration_images"  # Folder with checkerboard images
    try:
        camera_matrix, dist_coeffs, _, _ = calibrate_camera(calibration_folder)
        print("\nCamera Intrinsic Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        # Save calibration results
        np.savez("camera_calibration.npz", 
                camera_matrix=camera_matrix, 
                dist_coeffs=dist_coeffs)
        print("Calibration saved to camera_calibration.npz")
        
        # 2. Estimate depth scale
        # This requires an RGB-D image pair with a known size object
        try:
            rgb_image = cv2.imread("rgb_with_known_object.jpg")
            depth_image = cv2.imread("depth_with_known_object.png", -1)  # -1 to load as is
            
            if rgb_image is not None and depth_image is not None:
                depth_scale = estimate_depth_scale(
                    rgb_image, depth_image, camera_matrix, object_real_size=0.1
                )
                print(f"\nEstimated Depth Scale: {depth_scale}")
                print(f"To convert depth values to meters, multiply by {depth_scale}")
            else:
                print("\nDepth scale estimation requires RGB and depth images with a known object")
                
        except Exception as e:
            print(f"Error in depth scale estimation: {e}")
            
    except Exception as e:
        print(f"Error in camera calibration: {e}")
        print("\nTo use this script:")
        print("1. Create a folder 'calibration_images' with multiple images of a checkerboard")
        print("2. For depth scale, provide RGB and depth images with an object of known size")