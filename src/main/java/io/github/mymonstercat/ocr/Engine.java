package io.github.mymonstercat.ocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import cn.hutool.core.io.FileUtil;
import io.github.mymonstercat.ocr.rec.RecUtil;
import io.github.mymonstercat.ocr.util.ResultUtil;
import io.github.mymonstercat.ocr.util.MatUtil;
import io.github.mymonstercat.ocr.det.OperatorUtil;
import lombok.SneakyThrows;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.*;

import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Monster
 */
public class Engine {


    static final OrtEnvironment env = OrtEnvironment.getEnvironment();

    public static void main(String[] args) {
        runOcr();
    }

    public static void runOcr() {
        List<Mat> recMatList = det("src/main/resources/3.png");
        String rec = rec(recMatList);
        System.out.println(rec);

    }

    @SneakyThrows
    public static List<Mat> det(String imgPath) {
        long startTime = System.nanoTime();
        Mat mat = opencv_imgcodecs.imread(imgPath);
        MatUtil.BGR2RGB(mat);
        ArrayList<Mat> recMatList = new ArrayList<>();
        // rec
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setSessionLogVerbosityLevel(4);
            options.setCPUArenaAllocator(false);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            OrtSession detSession = env.createSession("src/main/resources/ch_PP-OCRv4_det_infer.onnx", options);

            Map<String, Object> transformed = OperatorUtil.ImgDetOperatorsLine(mat);

            Mat trsMat = (Mat) transformed.get("image");
            float[] floatData = (float[]) transformed.get("array");
            long[] shapeArray = new long[]{1, trsMat.channels(), trsMat.rows(), trsMat.cols()};

            Map<String, OnnxTensor> tensorMap = Map.of(detSession.getInputNames().iterator().next(), OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shapeArray));
            OrtSession.Result detResult = detSession.run(tensorMap);

            // 处理返回结果
            float[][][][] resultArray = (float[][][][]) detResult.get(0).getValue();
            float[] shape = (float[]) transformed.get("shape");
            float[][] shapeList = new float[1][4];
            shapeList[0] = shape;
            List<Point2f> point2fs = ResultUtil.process(resultArray, shapeList);


            int i = point2fs.size() - 1;
            while (i >= 0) {
                Point2f expandedPoints = point2fs.get(i);
                Point pt1 = new Point((int) expandedPoints.position(0).x(), (int) expandedPoints.position(0).y());
                Point pt3 = new Point((int) expandedPoints.position(2).x(), (int) expandedPoints.position(2).y());
                Mat apply = trsMat.apply(new Rect(pt1, pt3));
                recMatList.add(apply);
                i--;
            }

        }

        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        System.out.println("方法执行时间： " + (executionTime / 1_000_000) + " 毫秒");
        return recMatList;
    }

    @SneakyThrows
    public static String rec(List<Mat> recMatList) {
        StringBuilder sb = new StringBuilder();
        try (OrtSession recSession = env.createSession("src/main/resources/ch_PP-OCRv4_rec_infer.onnx")) {
            for (Mat mat : recMatList) {
                Mat clone = mat.clone();
                Map<String, OnnxTensor> tensorMap = Map.of(recSession.getInputNames().iterator().next(), RecUtil.img2OnnxTensor(env, clone));
                OrtSession.Result out = recSession.run(tensorMap);

                String cn = char2CN(out);
                sb.append(cn + "\n");
                clone.close();

            }
        }
        return sb.toString();

    }


    @SneakyThrows
    public static String char2CN(OrtSession.Result out) {

        List<String> charList = charList();
        ArrayList<Float> scoreList = new ArrayList<>();
        ArrayList<Integer> indexList = new ArrayList<>();
        Object value = out.get(0).getValue();
        float[][][] sd = (float[][][]) value;
        int row = sd[0].length;
        int col = sd[0][0].length;
        for (int i = 0; i < row; i++) {
            float max = sd[0][i][0];
            int index = 0;
            for (int j = 0; j < col; j++) {
                if (sd[0][i][j] > max) {
                    max = sd[0][i][j];
                    index = j;
                }
            }
            scoreList.add(max);
            indexList.add(index);
        }

        StringBuilder builder = new StringBuilder();
        for (Integer index : indexList) {
            if (index != 0) {
                builder.append(charList.get(index));
            }
        }
        return builder.toString();

    }

    public static List<String> charList() {
        String path = "/Users/monster/IdeaProjects/myself/ai-study/ort-java/src/main/resources/ppocr_keys_v1.txt";
        ArrayList<String> result = new ArrayList<>();
        result.add("blank");
        List<String> list = FileUtil.readLines(FileUtil.newFile(path), Charset.defaultCharset());
        result.addAll(list);
        result.add(" ");
        return result;
    }
}
