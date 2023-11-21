package io.github.mymonstercat.ocr.operator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KeepKeys implements IOperator {
    private List<String> keepKeys;

    public KeepKeys(List<String> keepKeys) {
        this.keepKeys = keepKeys;
    }

    @Override
    public Map<String, Object> apply(Map<String, Object> data) {
        Map<String, Object> result = new HashMap<>();
        for (String keepKey : keepKeys) {
            if (data.containsKey(keepKey)) {
                result.put(keepKey, data.get(keepKey));
            }
        }
        return result;
    }

}
