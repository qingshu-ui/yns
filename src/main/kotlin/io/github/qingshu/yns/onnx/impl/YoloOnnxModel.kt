package io.github.qingshu.yns.onnx.impl

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import io.github.qingshu.yns.onnx.AbstractDetect
import io.github.qingshu.yns.onnx.AbstractOnnxModel
import io.github.qingshu.yns.dto.DetectStatus
import io.github.qingshu.yns.dto.Detection
import io.github.qingshu.yns.onnx.utils.MatUtils
import nu.pattern.OpenCV
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import java.nio.FloatBuffer
import java.nio.file.Files

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
class YoloOnnxModel(
    modelPath: String,
    options: OrtSession.SessionOptions = OrtSession.SessionOptions(),
) : AbstractOnnxModel<YoloOnnxModel.Detect>(modelPath, options) {

    data class Detect(
        var mat: Mat,
        var label: List<String>,
        var conf: Float = 0.3f,
        var iou: Float = 0.5f,
        var output: List<Detection>? = null,
    ) : AbstractDetect()

    override fun preprocess(input: Detect): Detect {
        checkStatus(input.status, DetectStatus.PREPROCESS)
        val mat = input.mat
        val size = inputSize[inputNames.first()]!!
        // 1 letterbox
        val resizedMat = MatUtils.letterbox(mat, size)
        //val resizedMat = MatUtils.scaleByPadding(mat,size)

        // 2 normalize
        resizedMat.convertTo(resizedMat, CvType.CV_32FC3, 1.0 / 255)

        // 3 transpose
        val inputBuffer = FloatBuffer.allocate((size.height * size.width * resizedMat.channels()).toInt())
        MatUtils.hwc2chw(resizedMat, resizedMat)
        resizedMat.get(0, 0, inputBuffer.array())
        resizedMat.release()

        // 4 create onnx tensor
        val tensor = OnnxTensor.createTensor(env, inputBuffer, inputShape[0])
        input.tensors = mutableMapOf(inputNames.first() to tensor)
        input.status = DetectStatus.INFERENCE
        return input
    }

    @Suppress("UNCHECKED_CAST")
    override fun postprocess(result: Detect): Detect {
        checkStatus(result.status, DetectStatus.POSTPROCESS)
        // 1 matrix transpose
        val predictions = (result.result?.get(0)?.value as Array<Array<FloatArray>>)[0]
        val transposed = MatUtils.transposeMatrix(predictions)

        // 2 filter
        val class2Bbox = hashMapOf<Int, ArrayList<FloatArray>>()
        transposed.forEach { bbox ->
            val cond = bbox.copyOfRange(4, bbox.size)
            val label = MatUtils.maxIndex(cond)
            val conf = cond[label]
            if (conf < result.conf) return@forEach
            bbox[4] = conf
            MatUtils.rescaleByPadding(bbox, result.mat.size(), inputSize[inputNames.first()]!!)
            MatUtils.xywh2xyxy(bbox)
            if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) return@forEach
            class2Bbox.getOrPut(label) { arrayListOf() }.add(bbox)
        }

        // 3 non max suppress
        val detections = arrayListOf<Detection>()
        class2Bbox.forEach { (label, bboxList) ->
            val className = result.label[label]
            val nonMaxSuppression = MatUtils.nonMaxSuppression(bboxList, result.iou)
            nonMaxSuppression.forEach { bbox ->
                detections.add(
                    Detection(
                        label = className,
                        labelIndex = label,
                        bbox = bbox.copyOfRange(0, 4),
                        confidence = bbox[4]
                    )
                )
            }
        }
        result.output = detections
        result.status = DetectStatus.COMPLETED
        return result
    }

    fun detect(imagePath: String, labelPath: String, conf: Float = 0.3f, iou: Float = 0.5f): List<Detection> {
        val mat = Imgcodecs.imread(imagePath)
        val labelFile = File(labelPath)
        if (mat.empty() || !labelFile.exists()) {
            throw IllegalArgumentException("Empty mat")
        }
        return detect(mat, labelPath)
    }

    fun detect(mat: Mat, labelPath: String, conf: Float = 0.3f, iou: Float = 0.5f): List<Detection> {
        val labelFile = File(labelPath)
        if (mat.empty() || !labelFile.exists()) {
            throw IllegalArgumentException("Empty mat")
        }
        val label = Files.lines(labelFile.toPath()).toList()
        val detect = Detect(
            mat = mat,
            label = label,
            conf = conf,
            iou = iou
        )
        return detect(detect)
    }

    fun detect(detect: Detect): List<Detection> {
        val preprocess = preprocess(detect)
        val runInference = runInference(preprocess)
        val postprocess = postprocess(runInference)
        if(postprocess.status == DetectStatus.COMPLETED){
            return postprocess.use { postprocess.output ?: emptyList() }
        }
        throw RuntimeException("Detection error!")
    }
}

/* Example
fun main(vararg args: String) {
    OpenCV.loadLocally()
    val (modelPath, labelPath) = listOf(
        "d:/Users/17186/Downloads/Compressed/models/text-select/text-select-yolov8s.onnx",
        "d:/Users/17186/Downloads/Compressed/models/text-select/text-select-label.names"
    )
    val (img1, img2, img3, img4, img5) = listOf(
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/0.png",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/1.png",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/2.png",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/3.png",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/4.png",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/5.png",
    )
    val model = YoloOnnxModel(
        modelPath,
        OrtSession.SessionOptions().apply {
            this.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
    )
    model.use {
        val result = model.detect(img2, labelPath)
        println(result.size)
    }
}
 */