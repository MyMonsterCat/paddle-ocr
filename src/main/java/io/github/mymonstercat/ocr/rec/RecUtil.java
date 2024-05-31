package io.github.mymonstercat.ocr.rec;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import io.github.mymonstercat.ocr.util.MatUtil;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;

/**
 * @author Monster
 */
public class RecUtil {

    static final int imgHeight = 48;
    static final int imgWidth = 320;
    static final int imgChannel = 3;

    public static OnnxTensor img2OnnxTensor(OrtEnvironment environment, Mat mat) throws OrtException {
        MatUtil.BGR2RGB(mat);
        resizeImg(mat, imgHeight);
        MatUtil.normalization(mat);
        FloatBuffer floatBuffer = MatUtil.convertHWCtoCHW(mat);
        return OnnxTensor.createTensor(environment, floatBuffer, new long[]{1, 3, imgHeight, mat.cols()});
    }

    // 将图片调整至制定宽度
    private static void resizeImg(Mat mat, int normalHeight) {
        double ratio = (double) mat.cols() / mat.rows();
        int imgWidthCal = (int) (normalHeight * ratio);
        int resizeWidth = (int) (normalHeight * ratio) + 1;
        if (resizeWidth > imgWidthCal) {
            resizeWidth = imgWidthCal;
        }
        opencv_imgproc.resize(mat, mat, new Size(resizeWidth, imgHeight));
    }
}
