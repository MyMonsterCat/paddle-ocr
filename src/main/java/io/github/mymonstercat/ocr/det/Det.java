package io.github.mymonstercat.ocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import cn.hutool.core.io.resource.ResourceUtil;
import io.github.mymonstercat.ocr.operator.IOperator;
import io.github.mymonstercat.ocr.operator.OperatorFactory;
import io.github.mymonstercat.ocr.util.MatUtil;
import io.github.mymonstercat.ocr.util.OperatorUtil;
import lombok.SneakyThrows;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Monster
 */
public class Det {
    static final double thresh = 0.3;
    static final double box_thresh = 0.5;
    static final double max_candidates = 1000;
    static final double unclip_ratio = 1.6;
    static final boolean use_dilation = true;
    static final String score_mode = "fast";

    @SneakyThrows
    public static void main(String[] args) {

        String modelPath = ResourceUtil.getResource("ch_PP-OCRv4_det_infer.onnx").getPath();
        String imgPath = ResourceUtil.getResource("3.png").getPath();
        Mat mat = opencv_imgcodecs.imread(imgPath);
        opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_BGR2RGB);

        final OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setSessionLogVerbosityLevel(4);
            options.setCPUArenaAllocator(false);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            OrtSession session = env.createSession(modelPath, options);
            mat.convertTo(mat, opencv_core.CV_32FC3);
            Map<String, Map<String, Object>> opParamDict = new HashMap<>();
            opParamDict.put("DetResizeForTest", Map.of("limit_side_len", 736, "limit_type", "min"));
            opParamDict.put("KeepKeys", Map.of("keep_keys", List.of("image", "shape")));
            opParamDict.put("NormalizeImage", Map.of("mean", List.of(0.485, 0.456, 0.406), "order", "hwc", "scale", "1./255.", "std", List.of(0.229, 0.224, 0.225)));
            opParamDict.put("ToCHWImage", null);

            List<IOperator> operators = OperatorFactory.createOperators(opParamDict);
            HashMap<String, Object> inputData = new HashMap<>();
            inputData.put("image", mat);
            Map<String, Object> transformed = OperatorUtil.transform(inputData, operators);
            Mat trsMat = (Mat) transformed.get("image");
            long[] shapeArray = new long[]{1, trsMat.channels(), trsMat.rows(), trsMat.cols()};
            float[] floatData = MatUtil.matToArray(trsMat);
            OnnxTensor onnxTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shapeArray);
            Map<String, OnnxTensor> tensorMap = Map.of(session.getInputNames().iterator().next(), onnxTensor);

            long startTime = System.nanoTime();
            OrtSession.Result out = session.run(tensorMap);
            long endTime = System.nanoTime();
            long executionTime = endTime - startTime;
            System.out.println("方法执行时间： " + (executionTime / 1_000_000) + " 毫秒");

            // 处理返回结果


        }
    }




}
