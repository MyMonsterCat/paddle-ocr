package io.github.mymonstercat.ocr.det;

import java.util.Map;

public interface IOperator {

    int sort();
    Map<String, Object> apply(Map<String, Object> data);
}
