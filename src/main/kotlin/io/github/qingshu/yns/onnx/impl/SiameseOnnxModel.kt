package io.github.qingshu.yns.onnx.impl

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession.SessionOptions
import io.github.qingshu.yns.onnx.AbstractDetect
import io.github.qingshu.yns.onnx.AbstractOnnxModel
import io.github.qingshu.yns.dto.DetectStatus
import io.github.qingshu.yns.onnx.utils.MatUtils
import nu.pattern.OpenCV
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
class SiameseOnnxModel(
    modelPath: String,
    options: SessionOptions = SessionOptions(),
) : AbstractOnnxModel<SiameseOnnxModel.Detect>(modelPath, options) {

    data class Detect(
        val input0: Mat,
        val input1: Mat,
        var output: Float? = null,
    ) : AbstractDetect()

    override fun preprocess(input: Detect): Detect {
        checkStatus(input.status, DetectStatus.PREPROCESS)
        val (input0, input1) = input
        val (name0, name1) = inputNames
        val size0 = inputSize[name0]!!
        val size1 = inputSize[name1]!!

        // letterbox
        val resized0 = MatUtils.letterbox(input0, size0)
        val resized1 = MatUtils.letterbox(input1, size1)

        // normalize
        resized0.convertTo(resized0, CvType.CV_32FC3, 1.0 / 255)
        resized1.convertTo(resized1, CvType.CV_32FC3, 1.0 / 255)

        // transpose
        val inputBuffer0 = FloatBuffer.allocate((size0.height * size0.width * resized0.channels()).toInt())
        val inputBuffer1 = FloatBuffer.allocate((size1.height * size1.width * resized1.channels()).toInt())
        MatUtils.hwc2chw(resized0, resized0)
        MatUtils.hwc2chw(resized1, resized1)
        resized0.get(0, 0, inputBuffer0.array())
        resized1.get(0, 0, inputBuffer1.array())
        resized0.release()
        resized1.release()

        // create onnx tensor
        val tensor0 = OnnxTensor.createTensor(env, inputBuffer0, inputShape[0])
        val tensor1 = OnnxTensor.createTensor(env, inputBuffer1, inputShape[1])
        input.tensors = mutableMapOf(
            name0 to tensor0,
            name1 to tensor1,
        )
        input.status = DetectStatus.INFERENCE
        return input
    }

    @Suppress("unchecked_cast")
    override fun postprocess(result: Detect): Detect {
        checkStatus(result.status, DetectStatus.POSTPROCESS)
        val output = (result.result?.get(0)?.value as Array<FloatArray>)[0][0]
        result.output = sigmoid(output)
        result.status = DetectStatus.COMPLETED
        return result
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x.toDouble()).toFloat())
    }

    fun detect(mat0: Mat, mat1: Mat): Float {
        val detect = Detect(
            input0 = mat0,
            input1 = mat1,
        )
        return detect(detect)
    }

    fun detect(img0: String, img1: String): Float {
        val mat0 = Imgcodecs.imread(img0)
        val mat1 = Imgcodecs.imread(img1)
        if (mat0.empty() || mat1.empty()) {
            throw IllegalArgumentException("Mat is empty")
        }
        return detect(mat0, mat1)
    }

    fun detect(detect: Detect): Float {
        val preprocess = preprocess(detect)
        val runInference = runInference(preprocess)
        val postprocess = postprocess(runInference)
        if (postprocess.status == DetectStatus.COMPLETED) {
            return postprocess.use { postprocess.output!! }
        }
        throw RuntimeException("Detection error!")
    }
}

/* Example
fun main(args: Array<String>) {
    OpenCV.loadLocally()
    val input0 = listOf(
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/img/Angelic_01.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/img/Angelic_02.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/img/Atem_01.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/img/Atl_01.png",
    )
    val modelPath = "d:/Users/17186/Downloads/Compressed/models/text-select/siamese-model.onnx"
    val model = SiameseOnnxModel(modelPath, SessionOptions().apply {
        setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT)
    })

    model.use {
        val result = model.detect(input0[0], input0[1])
        println(result)
    }
}
 */