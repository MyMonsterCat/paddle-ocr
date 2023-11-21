package io.github.mymonstercat.ocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.resource.ResourceUtil;
import lombok.SneakyThrows;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Monster
 */
public class Rec {

    static final int imgHeight = 48;
    static final int imgWidth = 320;
    static final int imgChannel = 3;

    @SneakyThrows
    public static void main(String[] args) {

        String modelPath = "/Users/monster/IdeaProjects/myself/ai-study/ort-java/src/main/resources/ch_PP-OCRv4_rec_infer.onnx";
        String imgPath = ResourceUtil.getResource("img_2.png").getPath();
        Mat mat = opencv_imgcodecs.imread(imgPath);
        opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_BGR2RGB);

        final OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            OrtSession session = env.createSession(modelPath, options);

            double ratio = (double) mat.cols() / mat.rows();
            int imgWidthCal = (int) (imgHeight * ratio);
            int resizeWidth = (int) (imgHeight * ratio) + 1;
            if (resizeWidth > imgWidthCal) {
                resizeWidth = imgWidthCal;
            }
            opencv_imgproc.resize(mat, mat, new Size(resizeWidth, imgHeight));
            mat.convertTo(mat, opencv_core.CV_32FC3);
            int cols = mat.cols();
            float[] floatData = new float[3 * 48 * cols];


            for (int h = 0; h < 48; h++) {
                for (int w = 0; w < cols; w++) {
                    int index0 = 48 * cols * 0 + h * cols + w;
                    int index1 = 48 * cols * 1 + h * cols + w;
                    int index2 = 48 * cols * 2 + h * cols + w;
                    BytePointer ptr = mat.ptr(h, w);

                    float aFloat0 = mat.ptr(h, w).getFloat(0);
                    float aFloat1 = mat.ptr(h, w).getFloat(1);
                    float aFloat2 = mat.ptr(h, w).getFloat(2);

                    floatData[index0] = (aFloat0 / 255.0f - 0.5f) / 0.5f;
                    floatData[index1] = (aFloat1 / 255.0f - 0.5f) / 0.5f;
                    floatData[index2] = (aFloat2 / 255.0f - 0.5f) / 0.5f;
                }
            }


//            for (int k = 0; k < 3; k++) {
//                for (int i = 0; i < 48; i++) {
//                    for (int j = 0; j < cols; j++) {
//                        int index = (k * 48 + i) * cols + j;
//                        float aFloat = mat.ptr(i, j).getFloat(k);
//                        floatData[index] = (aFloat / 255.0f - 0.5f) / 0.5f;
////                        System.out.println(i + "---" + j + "---" + k + ">>>" + mat.ptr(i, j).getFloat(k) + ">>" + floatData[index]);
//                    }
//                }
//            }

            extracted(imgPath, floatData);

            long[] shape = {1, 3, 48, cols};
            OnnxTensor onnxTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatData), shape);
            Map<String, OnnxTensor> tensorMap = Map.of(session.getInputNames().iterator().next(), onnxTensor);

            long startTime = System.nanoTime();
            OrtSession.Result out = session.run(tensorMap);
            long endTime = System.nanoTime();
            long executionTime = endTime - startTime;
            System.out.println("方法执行时间： " + (executionTime / 1_000_000) + " 毫秒");

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
            System.out.println(builder);
        }


    }

    private static void extracted(String imgPath, float[] floatData) {
        // 文件路径
        String filePath = imgPath.replace("img_3.png", "a.txt");

        try {
            // 创建FileWriter对象并指定文件路径
            FileWriter fileWriter = new FileWriter(filePath);

            // 创建BufferedWriter对象，提高写入性能
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

            // 遍历float数组并将每个元素写入文件
            for (float value : floatData) {
                // 将float值转换为字符串并写入文件
                bufferedWriter.write(Float.toString(value));
                // 每个元素之间用换行符分隔
                bufferedWriter.newLine();
            }

            // 关闭BufferedWriter
            bufferedWriter.close();

            System.out.println("数据已成功写入文件 " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
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


    //            String filePath = ResourceUtil.getResource("final.txt").getPath();
//
//            // 读取文件并解析为 float 数组
//            float[] floatArray = readFloatArrayFromFile(filePath);
    private static float[] readFloatArrayFromFile(String filePath) {

        List<String> stringList = FileUtil.readLines(filePath, Charset.defaultCharset());
        float[] floatArray = new float[stringList.size()];
        for (int i = 0; i < stringList.size(); i++) {
            floatArray[i] = Float.parseFloat(stringList.get(i));
        }
        return floatArray;
    }

}
