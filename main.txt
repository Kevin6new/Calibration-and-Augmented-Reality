/* CS 5330 PATTERN RECOGNITION & COMPUTER VISION
PROJECT 4: CALIBRATION & AUGMENTATION
BASIL REJI & KEVIN SANI
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Function prototype for Task 6
void drawVirtualObject(Mat& framecpy, const Mat& rvec, const Mat& tvec, const Mat& cameraMatrix, const Mat& distortionCoefficients);

int main() {
    // Open default camera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera" << endl;
        return -1;
    }

    Mat frame;
    vector<Point2f> corners;
    Size boardSize(9, 6);  // Size of the checkerboard pattern
    vector<vector<Point2f>> corner_list;
    vector<vector<Vec3f>> point_list;
    vector<Vec3f> point_set;

    int image_counter = 0; // Counter for saving images
    bool task4_active = false;  // Flag to indicate if Task 4 is active
    bool harris_active = false; // Flag to indicate if Harris corner detection is active

    // Read camera calibration parameters from file
    FileStorage fs("intrinsic_parameters.yml", FileStorage::READ);
    Mat cameraMatrix, distortionCoefficients;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distortionCoefficients;
    fs.release();

    Mat rvec, tvec; // Rotation and translation vectors

    while (true) {
        cap >> frame;  // Capture frame from camera

        // Convert frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Harris Corner Detection when 'h' is pressed
        if (harris_active) {
            Mat corners, corner_norm;
            cornerHarris(gray, corners, 2, 3, 0.04);

            // Normalize corner response
            normalize(corners, corner_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

            // Draw circles around detected corners
            for (int i = 0; i < corner_norm.rows; i++) {
                for (int j = 0; j < corner_norm.cols; j++) {
                    if ((int)corner_norm.at<float>(i, j) > 100) { // Threshold for corner detection
                        circle(frame, Point(j, i), 5, Scalar(0, 255, 0), 2, 8, 0);
                    }
                }
            }
        }

        // Find chessboard corners
        bool found = findChessboardCorners(gray, boardSize, corners);

        if (found) {
            // Refine corner positions
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // Draw corners on the frame
            drawChessboardCorners(frame, boardSize, Mat(corners), found);

            // Print number of corners and coordinates of the first corner
            cout << "Number of corners: " << corners.size() << endl;
            cout << "Coordinates of first corner: (" << corners[0].x << ", " << corners[0].y << ")" << endl;

            // Store the corners and corresponding 3D world points
            corner_list.push_back(corners);
            point_set.clear();
            for (int i = 0; i < boardSize.height; ++i) {
                for (int j = 0; j < boardSize.width; ++j) {
                    point_set.push_back(Vec3f(j, i, 0)); // Assuming chessboard is on the xy plane
                }
            }
            point_list.push_back(point_set);

            // Check if enough calibration frames are selected
            if (corner_list.size() >= 5) {
                // Run camera calibration
                solvePnP(Mat(point_set), Mat(corners), cameraMatrix, distortionCoefficients, rvec, tvec);

                // Project 3D points onto the image plane
                vector<Point2f> projected_points;
                projectPoints(point_set, rvec, tvec, cameraMatrix, distortionCoefficients, projected_points);

                // Draw projected points or 3D axes on the frame
                for (int i = 0; i < projected_points.size(); ++i) {
                    circle(frame, projected_points[i], 5, Scalar(0, 255, 0), -1); // Draw projected points
                }

                // Display the frame with projected points or 3D axes
                imshow("Frame", frame);

                if (task4_active) {
                    // Print rotation and translation data
                    cout << "Rotation vector:" << endl << rvec << endl;
                    cout << "Translation vector:" << endl << tvec << endl;
                }
            }
        }

        // Task 4 activation when 'o' is pressed
        if (waitKey(1) == 'o') {
            task4_active = true;
        }

        // Harris Corner Detection activation when 'h' is pressed
        if (waitKey(1) == 'h') {
            harris_active = true;
        }

        // Break loop if ESC key is pressed
        if (waitKey(1) == 27) {
            break;
        }
    }

    cap.release();  // Release the camera
    destroyAllWindows();  // Close all OpenCV windows

    return 0;
}
