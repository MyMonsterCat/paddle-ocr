package io.github.mymonstercat.ocr;

import io.github.mymonstercat.ocr.operator.IOperator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class OperatorUtil {
    public static Map<String, Object> transform(Map<String, Object> data, List<IOperator> ops) {
        if (ops == null) {
            ops = new ArrayList<>();
        }

        for (IOperator op : ops) {
            data = op.apply(data);
            if (data == null) {
                return null;
            }
        }
        return data;
    }
}
