package io.github.mymonstercat.ocr.det;

import io.github.mymonstercat.ocr.det.operator.DetResizeForTest;
import io.github.mymonstercat.ocr.det.operator.KeepKeys;
import io.github.mymonstercat.ocr.det.operator.NormalizeImage;
import io.github.mymonstercat.ocr.det.operator.ToCHWImage;

import java.util.*;


public class OperatorFactory {
    public static List<IOperator> createOperators(Map<String, Map<String, Object>> opParamDict) {
        List<IOperator> ops = new ArrayList<>();

        for (Map.Entry<String, Map<String, Object>> entry : opParamDict.entrySet()) {
            String opName = entry.getKey();
            Map<String, Object> param = entry.getValue();

            if (param == null) {
                param = new HashMap<>();
            }

            IOperator op = createOperator(opName, param);
            ops.add(op);
        }

        return ops;
    }

    private static IOperator createOperator(String opName, Map<String, Object> param) {
        switch (opName) {
            case "DetResizeForTest":
                return new DetResizeForTest((int) param.get("limit_side_len"), (String) param.get("limit_type"));
            case "KeepKeys":
                return new KeepKeys((List<String>) param.get("keep_keys"));
            case "NormalizeImage":
                return new NormalizeImage((List<Double>) param.get("mean"), (String) param.get("order"), (String) param.get("scale"), (List<Double>) param.get("std"));
            case "ToCHWImage":
                return new ToCHWImage();
            default:
                throw new IllegalArgumentException("Unsupported operator: " + opName);
        }
    }
}









