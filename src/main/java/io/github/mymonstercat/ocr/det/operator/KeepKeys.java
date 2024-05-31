package io.github.mymonstercat.ocr.det.operator;

import io.github.mymonstercat.ocr.det.IOperator;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KeepKeys implements IOperator {
    private List<String> keepKeys;

    public KeepKeys(List<String> keepKeys) {
        this.keepKeys = keepKeys;
    }

    @Override
    public int sort() {
        return 4;
    }

    @Override
    public Map<String, Object> apply(Map<String, Object> data) {
        Map<String, Object> result = new HashMap<>();
        for (String keepKey : keepKeys) {
            if (data.containsKey(keepKey)) {
                result.put(keepKey, data.get(keepKey));
            }
        }
        Mat trsMat = (Mat) result.get("image");
        return result;
    }

}
