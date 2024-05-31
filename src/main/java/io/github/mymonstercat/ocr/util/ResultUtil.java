package io.github.mymonstercat.ocr.util;


import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;

import java.util.ArrayList;
import java.util.List;

public class ResultUtil {
    static double thresh = 0.3;
    static double boxThresh = 0.5;
    static double maxCandidates = 1000;
    static double unClipRatio = 1.6;
    static boolean useDilation = false;
    static String scoreMode = "fast";
    static Mat dilationKernel = Mat.ones(2, 2, opencv_core.CV_8UC1).asMat();


    public static List<Point2f> process(float[][][][] data, float[][] shapeArray) {
        List<Point2f> pointResult = new ArrayList<>();
        float[][] pred = data[0][0];
        int height = pred.length;
        int width = pred[0].length;

        int[][] segmentation = new int[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (pred[i][j] > thresh) {
                    segmentation[i][j] = 1;
                } else {
                    segmentation[i][j] = 0;
                }

            }
        }
        for (int i = 0; i < data[0].length; i++) {
            float[] shape = shapeArray[i];
            float srcHeight = shape[0];
            float srcWidth = shape[1];
            Mat bitmap = new Mat();

            if (dilationKernel != null) {
                Mat srcMat = convertIntArrayToMat(segmentation);
                srcMat.convertTo(srcMat, opencv_core.CV_8UC1);
                opencv_imgproc.dilate(srcMat, bitmap, dilationKernel);
            } else {
                System.out.println("error");
            }

            Mat outs = new Mat();
            MatVector contours = new MatVector();
            opencv_core.multiply(bitmap, 255.0);
            bitmap.convertTo(bitmap, opencv_core.CV_8UC1);
            opencv_imgproc.findContours(bitmap, contours, outs, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_SIMPLE);

            for (int j = 0; j < contours.size(); j++) {
                Mat mat = contours.get(j);
                RotatedRect minAreaRect = opencv_imgproc.minAreaRect(mat);
                // 获取矩形的四个角点
                Point2f points = new Point2f(4);
                minAreaRect.points(points);

                // 计算矩形的中心
                Point2f center = minAreaRect.center();

                // 扩展比例
                float unclipRatio = 2.5f;

                // 计算扩展后的角点
                Point2f expandedPoints = new Point2f(4);
                for (int m = 0; m < 4; m++) {
                    float x = points.position(m).x() - center.x();
                    float y = points.position(m).y() - center.y();
                    // 根据扩展比例调整x和y
                    x *= 1.05f;
                    y *= unclipRatio;
                    // 将调整后的向量加回到中心点
                    expandedPoints.position(m).x(center.x() + x).y(center.y() + y);
                }
                pointResult.add(expandedPoints);
            }
        }

        return pointResult;
    }


    public static Mat convertIntArrayToMat(int[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        Mat mat = new Mat(rows, cols, opencv_core.CV_32S);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mat.ptr(i, j).putInt(array[i][j]);
            }
        }
        return mat;
    }

    public static void drawOnImg(Mat mat, List<Point2f> point2fs, String savePath) {
        for (Point2f expandedPoints : point2fs) {
            for (int k = 0; k < 4; k++) {
                Point pt1 = new Point((int) expandedPoints.position(k).x(), (int) expandedPoints.position(k).y());
                Point pt2 = new Point((int) expandedPoints.position((k + 1) % 4).x(), (int) expandedPoints.position((k + 1) % 4).y());
                opencv_imgproc.line(mat, pt1, pt2, new Scalar(0, 255, 0, 0));
            }
        }
        opencv_imgcodecs.imwrite(savePath, mat);
    }

}
