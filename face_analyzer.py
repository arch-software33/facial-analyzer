import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class FaceAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Camera matrix estimation (can be calibrated for better accuracy)
        self.focal_length = 1000.0
        self.center = (640 / 2, 480 / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.center[0]],
            [0, self.focal_length, self.center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process a single frame and return the annotated frame and facial metrics.
        
        Args:
            frame: Input frame from video stream
            
        Returns:
            Tuple containing:
            - Annotated frame
            - Dictionary with facial metrics (or None if no face detected)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame, None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Extract key points for pose estimation
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),    # Nose tip
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # Chin
            (face_landmarks.landmark[226].x * w, face_landmarks.landmark[226].y * h), # Left eye corner
            (face_landmarks.landmark[446].x * w, face_landmarks.landmark[446].y * h), # Right eye corner
            (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h),   # Left mouth corner
            (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h)  # Right mouth corner
        ], dtype=np.float64)

        # Calculate head pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        if not success:
            return frame, None

        # Convert rotation vector to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = [float(angle) for angle in euler_angles]

        # Calculate eye positions and gaze
        left_eye_center = np.mean([
            [face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h],
            [face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h]
        ], axis=0)
        
        right_eye_center = np.mean([
            [face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h],
            [face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h]
        ], axis=0)

        # Draw facial landmarks
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Draw head pose axes
        axis_length = 50
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, axis_length)]), rotation_vec, translation_vec, 
            self.camera_matrix, self.dist_coeffs)
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # Add text annotations
        metrics = {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'left_eye': left_eye_center,
            'right_eye': right_eye_center
        }

        # Draw metrics on frame
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, metrics

    def run_video_capture(self):
        """
        Run real-time video capture and analysis.
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame, metrics = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Face Analysis', annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    analyzer.run_video_capture() 