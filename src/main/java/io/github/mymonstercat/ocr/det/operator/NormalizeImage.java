package io.github.mymonstercat.ocr.det.operator;

import io.github.mymonstercat.ocr.det.IOperator;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;
import java.util.Map;

public class NormalizeImage implements IOperator {
    private List<Double> mean;
    private String order;
    private String scale;
    private List<Double> std;

    public NormalizeImage(List<Double> mean, String order, String scale, List<Double> std) {
        this.mean = mean;
        this.order = order;
        this.scale = scale;
        this.std = std;
    }

    @Override
    public int sort() {
        return 2;
    }

    @Override
    public Map<String, Object> apply(Map<String, Object> data) {

        // 参数维度变换
        float means[] = {0.485f, 0.456f, 0.406f};
        float stds[] = {0.229f, 0.224f, 0.225f};
        float[][][] meanArray = new float[1][1][3];
        float[][][] stdArray = new float[1][1][3];
        for (int i = 0; i < 3; i++) {
            meanArray[0][0][i] = means[i];
            stdArray[0][0][i] = stds[i];
        }

        Mat mat = (Mat) data.get("image");
        mat.convertTo(mat, opencv_core.CV_32FC3);

        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();

        float[] floatData = new float[channels * height * width];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int i = 0; i < channels; i++) {
                    int index = height * width * i + h * width + w;
                    float value = mat.ptr(h, w).getFloat(i);
                    floatData[index] = (value / 255.0f - means[i]) / stds[i];
                }
            }
        }
        data.put("array", floatData);
        return data;
    }
}
