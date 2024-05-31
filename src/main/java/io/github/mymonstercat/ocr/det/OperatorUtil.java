package io.github.mymonstercat.ocr.det;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.*;

public class OperatorUtil {

    public static Map<String, Object> ImgDetOperatorsLine(Mat mat) {
        Map<String, Map<String, Object>> opParamDict = new HashMap<>();
        opParamDict.put("DetResizeForTest", Map.of("limit_side_len", 736, "limit_type", "min"));
        //opParamDict.put("KeepKeys", Map.of("keep_keys", List.of("image", "shape")));
        opParamDict.put("NormalizeImage", Map.of("mean", List.of(0.485, 0.456, 0.406), "order", "hwc", "scale", "1./255.", "std", List.of(0.229, 0.224, 0.225)));
        //opParamDict.put("ToCHWImage", null);
        List<IOperator> operators = OperatorFactory.createOperators(opParamDict);

        HashMap<String, Object> inputData = new HashMap<>();
        inputData.put("image", mat);
        Map<String, Object> transformed = transform(inputData, operators);
        return transformed;
    }


    public static Map<String, Object> transform(Map<String, Object> data, List<IOperator> ops) {
        if (ops == null) {
            ops = new ArrayList<>();
        }
        ops.sort(Comparator.comparingInt(IOperator::sort));
        for (IOperator op : ops) {
            data = op.apply(data);
            if (data == null) {
                return null;
            }
        }
        return data;
    }
}
