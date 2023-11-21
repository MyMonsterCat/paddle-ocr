package io.github.mymonstercat.ocr.operator;

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
    public Map<String, Object> apply(Map<String, Object> data) {
        Mat mat = (Mat) data.get("image");

        int width = mat.cols();
        int height = mat.rows();
        float means[] = {0.485f, 0.456f, 0.406f};
        float stds[] = {0.229f, 0.224f, 0.225f};

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int i = 0; i < 3; i++) {
                    float value = mat.ptr(h, w).getFloat(i);
                    float v = value * 1.0f / 255.0f - means[i] / stds[i];
                    mat.ptr(h, w).putFloat(v);
                }
            }
        }
        data.put("image", mat);
        return data;
    }
}
