#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string.h>
#include <unistd.h>

using namespace std;
using namespace cv;

Mat frame;
Mat bord_frame;

int main()
{
    float dst_size = 240.0;
    int wind_size = 40;
    Mat dst_bin_clone, window;

    VideoCapture cap("solidYellowLeft.mp4");
//    VideoCapture cap("video.mp4");

    if(!cap.isOpened()) {
        cout << "Error" << endl;
        return -1;
    }

    namedWindow("Bird", WINDOW_AUTOSIZE); //Params
    int WidthUpPlane = 72;
    createTrackbar("Width up plane", "Bird", &WidthUpPlane, 100);
    int HeigthUpPlane = 77;
    createTrackbar("Heigth up plane", "Bird", &HeigthUpPlane, 100);
    int WidthLowPlane = 88;
    createTrackbar("Width low plane", "Bird", &WidthLowPlane, 100);
    int HeigthLowPlane = 91;
    createTrackbar("Heigth low plane", "Bird", &HeigthLowPlane, 100);

    //Корректировка inRange
    ////////////////////////////////////////////////////////////////////
    namedWindow("Debug");
    int Scal0_1 = 0;
    createTrackbar("Scal0_1", "Debug", &Scal0_1, 255);
    int Scal0_2 = 0;
    createTrackbar("Scal0_2", "Debug", &Scal0_2, 255);
    int Scal0_3 = 32;
    createTrackbar("Scal0_3", "Debug", &Scal0_3, 255);

    int Scal1_1 = 233;
    createTrackbar("Scal1_1", "Debug", &Scal1_1, 255);
    int Scal1_2 = 255;
    createTrackbar("Scal1_2", "Debug", &Scal1_2, 255);
    int Scal1_3 = 255;
    createTrackbar("Scal1_3", "Debug", &Scal1_3, 255);
    ////////////////////////////////////////////////////////////////////

    while(1) {

        cap >> frame;

        if(frame.empty()) {
//            cout << "Frame empty" << endl;
//            break;
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        float rows = frame.rows;
        float cols = frame.cols;

        float p4_x = cols/100.0*WidthLowPlane;
        float p43_y = rows/100.0*HeigthLowPlane;
        float p3_x = cols/100.0*(100.0 - WidthLowPlane);

        float p1_x = cols/100.0*WidthUpPlane;
        float p12_y = rows/100.0*HeigthUpPlane;
        float p2_x = cols/100.0*(100.0 - WidthUpPlane);

        vector<Point2f> dst_bin_points;

        //параметры трапеции, которая рисуется на видео
        vector<Point> polyline;
        polyline.push_back(Point(p1_x, p12_y));
        polyline.push_back(Point(p2_x, p12_y));
        polyline.push_back(Point(p3_x, p43_y));
        polyline.push_back(Point(p4_x, p43_y));

        //параметры трапеции, которая используется для формирования матрицы перехода
        vector<Point2f> points;
        points.push_back(Point2f(p1_x, p12_y));
        points.push_back(Point2f(p2_x, p12_y));
        points.push_back(Point2f(p3_x, p43_y));
        points.push_back(Point2f(p4_x, p43_y));

        //параметры изображения, к которому будем конвертировать
        vector<Point2f> dst_points;
        dst_points.push_back(Point2f(dst_size, 0.0));
        dst_points.push_back(Point2f(0.0, 0.0));
        dst_points.push_back(Point2f(0.0, dst_size));
        dst_points.push_back(Point2f(dst_size, dst_size));

        double fps = cap.get(CAP_PROP_FPS);
        int time_mls = cap.get(CAP_PROP_POS_MSEC);
//        cout << "FPS: " << fps << " Time(mls): " << time_mls << endl;
        Mat frame4poly = frame.clone();         //копия изображения для отрисовки тапеции, фепесов, времени
        string fps_str = to_string(fps);        //фепесы
        string time_str = to_string(time_mls);  //время
        string info = "FPS: " + fps_str + " Time(mls): " + time_str;
        putText(frame4poly, info, Point(5, 100), 1, 2.0, Scalar(255, 255, 255));
        polylines(frame4poly, polyline, 1, Scalar(255, 0, 0), 4);

        //матрица перехода
        Mat Matrix = getPerspectiveTransform(points, dst_points);
        Mat dst;
        warpPerspective(frame, dst, Matrix, Size(240,240), INTER_LINEAR, BORDER_CONSTANT); //получение вида сверху

        //смена цвета BGR на (HSV||HLS) для дальнейшей бинаризации изображения dst
        Mat dst2NC = dst.clone();
        cvtColor(dst2NC, dst2NC, COLOR_BGR2HLS);
        Mat dst_bin = dst2NC.clone();
        inRange(dst_bin, Scalar(Scal0_1, Scal0_2, Scal0_3), Scalar(Scal1_1, Scal1_2, Scal1_3), dst_bin);

        dst_bin_clone = dst_bin.clone();
        for(int i = 0; i < dst_size/wind_size; i++) {
            for(int j = 0; j < 2*dst_size/wind_size - 1; j++) {
                Rect rect(j*wind_size/2, i*wind_size, wind_size, wind_size);
                window = dst_bin(rect);
                Moments mom = moments(window, true);
                if(mom.m00 > 100) {
                    Point2f point(j*wind_size/2 + float(mom.m10/mom.m00), i*wind_size + float(mom.m01/mom.m00));

                    bool dublicate = false;
                    for(size_t n = 0; n < dst_bin_points.size(); n++) {
                        if(norm(dst_bin_points[n] - point) < 10) {
                            dublicate = true;
                        }
                    }
                    if(!dublicate) {
                        dst_bin_points.push_back(point);
                        circle(dst_bin_clone, point, 5, Scalar(128), -1);
                    }
                }
            }
        }

        if(dst_bin_points.size() > 0) {
            vector<Point2f> frame_points;
            perspectiveTransform(dst_bin_points, frame_points, Matrix.inv());
            for(size_t i = 0; i < dst_bin_points.size(); i++) {
                circle(frame4poly, frame_points[i], 5, Scalar(0, 0, 0), -1);
            }
        }

        imshow("Frame", frame4poly);
        imshow("Bird", dst);
        imshow("NC", dst2NC);
        imshow("Debug", dst_bin);
        imshow("Dst bin debug", dst_bin_clone);


        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
