package io.github.mymonstercat.ocr.operator;

import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.HashMap;
import java.util.Map;

public class DetResizeForTest implements IOperator {
    private int resizeType = 0;
    private int limitSideLen;
    private String limitType;

    public DetResizeForTest(int limitSideLen, String limitType) {
        this.limitSideLen = limitSideLen;
        this.limitType = limitType;
    }

    @Override
    public Map<String, Object> apply(Map<String, Object> data) {
        Mat imgMat = (Mat) data.get("image");
        HashMap<String, Object> result = new HashMap<>();
        float ratioH = 0;
        float ratioW = 0;
        int width = 0;
        int height = 0;
        if (resizeType == 0) {
            width = imgMat.cols();
            height = imgMat.rows();
            float ratio = getRatio(height, width);

            int resizeH = (int) (height * ratio);
            int resizeW = (int) (width * ratio);

            resizeH = (int) (Math.round(resizeH / 32.0) * 32);
            resizeW = (int) (Math.round(resizeW / 32.0) * 32);

            try {
                if (resizeW <= 0 || resizeH <= 0) {
                    return null;
                }
                Mat resizedImg = new Mat();
                opencv_imgproc.resize(imgMat, resizedImg, new Size(resizeW, resizeH));
                result.put("image", resizedImg);

            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }

            ratioH = resizeH / (float) height;
            ratioW = resizeW / (float) width;
        }
        float[] shapeArray = {height, width, ratioH, ratioW};
        result.put("shape", shapeArray);
        return result;
    }

    private float getRatio(int height, int width) {
        float ratio;
        if ("max".equals(limitType)) {
            if (Math.max(height, width) > limitSideLen) {
                if (height > width) {
                    ratio = (float) limitSideLen / height;
                } else {
                    ratio = (float) limitSideLen / width;
                }
            } else {
                ratio = 1.0f;
            }
        } else {
            if (Math.min(height, width) < limitSideLen) {
                if (height < width) {
                    ratio = (float) limitSideLen / height;
                } else {
                    ratio = (float) limitSideLen / width;
                }
            } else {
                ratio = 1.0f;
            }
        }
        return ratio;
    }
}
