package io.github.mymonstercat.ocr.det.operator;

import io.github.mymonstercat.ocr.det.IOperator;

import java.util.Map;

public class ToCHWImage implements IOperator {

    @Override
    public int sort() {
        return 3;
    }

    @Override
    public Map<String, Object> apply(Map<String, Object> data) {
        return data;
    }

}
