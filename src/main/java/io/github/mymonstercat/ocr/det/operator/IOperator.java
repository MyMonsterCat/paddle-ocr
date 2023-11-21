package io.github.mymonstercat.ocr.operator;

import java.util.Map;

public interface IOperator {
    Map<String, Object> apply(Map<String, Object> data);
}
