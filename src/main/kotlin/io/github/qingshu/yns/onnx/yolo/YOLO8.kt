package io.github.qingshu.yns.onnx.yolo

import ai.onnxruntime.*
import io.github.qingshu.yns.onnx.utils.MatUtils.maxIndex
import io.github.qingshu.yns.onnx.utils.MatUtils.nonMaxSuppression
import io.github.qingshu.yns.onnx.utils.MatUtils.rescaleByPadding
import io.github.qingshu.yns.onnx.utils.MatUtils.scaleByPadding
import io.github.qingshu.yns.onnx.utils.MatUtils.transposeMatrix
import io.github.qingshu.yns.onnx.utils.MatUtils.whc2cwh
import io.github.qingshu.yns.onnx.utils.MatUtils.xywh2xyxy
import nu.pattern.OpenCV
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.slf4j.LoggerFactory
import java.io.BufferedReader
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer
import java.nio.file.Files
import java.util.stream.Collectors

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the example-ayaka project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
class YOLO8 : AutoCloseable {

    private lateinit var env: OrtEnvironment

    private lateinit var session: OrtSession

    private lateinit var inputShape: LongArray

    private lateinit var inputName: String

    private lateinit var labelNames: List<String>

    /**
     * 置信度（Confidence）
     */
    private var conf: Float = 0.25f

    /**
     * 交并比（Intersection over Union, IoU）
     */
    private var iou: Float = 0.5f

    /**
     * 原始图片的大小，记录用于还原边界框
     */
    private var rawSize: Size = Size()

    /**
     * 模型输入的大小
     */
    private var inputSize: Size = Size()


    init {
        OpenCV.loadLocally()
    }

    fun initializeModel(model: ByteArray) {
        env = OrtEnvironment.getEnvironment()

        try {
            session = env.createSession(model, OrtSession.SessionOptions())
        } catch (e: OrtException) {
            log.error("Could not load model, Because of ${e.message}")
        }

        // 获取模型输入名称
        this.inputName = session.inputNames.first()
        log.info("Loaded model, input name: $inputName")
        // 获取模型输入张量信息
        val tensorInfo = session.inputInfo[inputName]?.info as TensorInfo
        // 获取模型输入形状
        this.inputShape = tensorInfo.shape
        // 获取模型输入大小
        this.inputSize.width = inputShape[2].toDouble()
        this.inputSize.height = inputShape[3].toDouble()
        log.info("Model input shape: ${inputSize.width} * ${inputSize.height}")
    }

    fun initializeLabel(reader: BufferedReader) {
        try {
            reader.use {
                labelNames = reader.lines().map(String::trim).collect(Collectors.toList())
            }
        } catch (e: IOException) {
            log.error("Could not load label, because of ${e.message}")
        }
    }

    /**
     * 在图片中检测目标
     * @param mat [Mat]
     * @param conf [Float] 置信度， 默认 0.25
     * @param iou [Float] 交并比，用来处理边界框重叠的参数
     * @return result [ArrayList]
     */
    @Synchronized
    fun detectObject(mat: Mat, conf: Float = 0.25f, iou: Float = 0.5f): ArrayList<Detection> {
        this.conf = conf
        this.iou = iou
        // 1. 前处理
        val input = prepareInput(mat)

        // 2. 使用模型推理
        val result = inference(input)

        // 3. 后处理
        val output = processOutput(result)

        // 4. 返回结果
        return output
    }

    /**
     * 在图片中检测目标
     * @param mat [Mat]
     * @param conf [Float] 置信度， 默认 0.25
     * @param iou [Float] 交并比，用来处理边界框重叠的参数
     * @return result [ArrayList]
     */
    fun detectObject(imagePath: String, conf: Float = 0.25f, iou: Float = 0.5f): ArrayList<Detection> {
        val imgMat = Imgcodecs.imread(imagePath)
        if (imgMat.empty()) {
            log.error("Could not open image, because image is empty")
            return ArrayList()
        }
        return detectObject(imgMat, conf, iou)
    }

    /**
     * 前处理
     * @param mat [Mat]
     * @return result [Map]
     */
    private fun prepareInput(mat: Mat): Map<String, OnnxTensor> {
        rawSize.width = mat.width().toDouble()
        rawSize.height = mat.height().toDouble()

        // 图片变换以匹配模型需要的大小
        val resizedMat = scaleByPadding(mat, inputSize)
        lateinit var whcArr: FloatArray
        try {
            // 归一化
            resizedMat.convertTo(resizedMat, CvType.CV_32FC3, 1.0 / 255)

            // 创建输入张量
            whcArr = FloatArray((inputSize.width * inputSize.height * mat.channels()).toInt())
            resizedMat.get(0, 0, whcArr)
        }finally {
            resizedMat.release()
        }
        // 张量变换：whc to chw
        val inputBuffer = FloatBuffer.wrap(whc2cwh(whcArr))
        val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

        return mapOf(inputName to inputTensor)
    }

    /**
     * 推理
     * @param input [Map]
     * @return [Array]
     */
    @Suppress("UNCHECKED_CAST")
    private fun inference(input: Map<String, OnnxTensor>): Array<FloatArray> {
        return session.run(input).use { result ->
                (result[0].value as Array<Array<FloatArray>>)[0]
            }
    }

    /**
     * 后处理
     * @param predictions
     * @return [ArrayList]
     */
    private fun processOutput(predictions: Array<FloatArray>): ArrayList<Detection> {
        // 1. 矩阵转置
        val transposedMatrix = transposeMatrix(predictions)
        // 2. 创建容器，用来接收符合置信度的结果
        val class2Bbox = hashMapOf<Int, ArrayList<FloatArray>>()
        // 3. 遍历
        for (bbox in transposedMatrix) {
            // 前四个是边界框值，后面是每个标签的概率
            val cond = bbox.copyOfRange(4, bbox.size)
            // 获取概率最高的标签(index)
            val label = maxIndex(cond)
            // 获取置信度（概率）
            val pConf = cond[label]
            // 如果不符合预期的置信度
            if (pConf < this.conf) continue
            bbox[4] = pConf
            // 映射到原始的坐标
            rescaleByPadding(bbox, rawSize, inputSize)
            // 转换边界框格式
            xywh2xyxy(bbox)
            // 简单的判断边界框是否超出图片范围
            if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) continue
            // 将符合的结果添加的容器
            class2Bbox.getOrPut(label) { arrayListOf() }.add(bbox)
        }

        // 至此，所有符合置信度conf的结果都在 class2Bbox 集合中
        // 创建新的集合，用来存放最终的检测结果
        val detectionList = arrayListOf<Detection>()
        for ((label, bboxList) in class2Bbox) {
            // 非极大值抑制，用于剔除边界框重叠，保留置信度最大的
            val nmxBoxes = nonMaxSuppression(bboxList, iou)
            for (bbox in nmxBoxes) {
                // 获取标签名
                val labelString = labelNames.getOrElse(label) { "Unknown" }
                // 将识别的结果添加到列表
                detectionList.add(
                    Detection(
                        label = labelString, labelIndex = label, bbox = bbox.copyOfRange(0, 4), confidence = bbox[4]
                    )
                )
            }
        }
        // 返回最终的结果
        return detectionList
    }

    override fun close() {
        try {
            session.close()
            env.close()
            // log.info("ONNX resources has been closed.")
        } catch (e: IOException) {
            log.error("Error closing ONNX resources: ${e.message}")
        }
    }

    companion object {
        private val log = LoggerFactory.getLogger(YOLO8::class.java)

        private fun newInstance(model: ByteArray, reader: BufferedReader): YOLO8 {
            val yolo = YOLO8()
            yolo.initializeModel(model)
            yolo.initializeLabel(reader)
            return yolo
        }

        /**
         * 从系统路径中加载模型
         * @param modelPath [String]
         * @param labelPath [String]
         */
        fun newInstance(modelPath: String, labelPath: String): YOLO8 {
            val modelFile = File(modelPath)
            val labelFile = File(labelPath)
            if (!labelFile.exists() && !modelFile.exists()) {
                log.error("$modelPath or $labelPath does not exist")
                throw IllegalArgumentException("$modelPath or $labelPath does not exist")
            }

            val modelByteArray = Files.readAllBytes(modelFile.toPath())
            val labelBufferedReader = Files.newBufferedReader(labelFile.toPath())
            return newInstance(modelByteArray, labelBufferedReader)
        }
    }
}