package io.github.qingshu.yns.onnx

import ai.onnxruntime.*
import io.github.qingshu.yns.annotation.Slf4j
import io.github.qingshu.yns.annotation.Slf4j.Companion.log
import io.github.qingshu.yns.exception.UnloadedModelException
import io.github.qingshu.yns.onnx.utils.MatUtils.hwc2chw
import io.github.qingshu.yns.onnx.utils.MatUtils.letterbox
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.IOException
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the example-ayaka project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
@Slf4j
class Siamese : ONNX {

    private val env = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private lateinit var inputSize: List<Size>
    private lateinit var inputShape: List<LongArray>
    private lateinit var inputName: List<String>
    private var load = false

    init {
        nu.pattern.OpenCV.loadLocally()
    }

    private fun initModel() {
        inputName = session.inputNames.toList()
        inputShape = session.inputInfo.run {
            listOf(
                (this[inputName[0]]?.info as TensorInfo).shape,
                (this[inputName[1]]?.info as TensorInfo).shape,
            )
        }
        inputSize = listOf(
            Size(inputShape[0][2].toDouble(), inputShape[0][3].toDouble()),
            Size(inputShape[1][2].toDouble(), inputShape[1][3].toDouble()),
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun detectImage(image0: Mat, image1: Mat): Float {
        cvtColor(image0)
        cvtColor(image1)

        val (resizeImg0, resizeImg1) = listOf(
            letterbox(image0, inputSize[0]),
            letterbox(image1, inputSize[1])
        )

        val (normalized0, normalized1) = processInput(resizeImg0) to processInput(resizeImg1)

        val inputTensors = image2Tensor(normalized0, normalized1)

        val result = session.run(inputTensors)
        val output = (result[0].value as Array<FloatArray>)[0][0]
        val conf = sigmoid(output)
        return conf
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x.toDouble()).toFloat())
    }

    private fun image2Tensor(mat0: Mat, mat1: Mat): Map<String, OnnxTensor> {
        val (fb0, fb1) = listOf(
            FloatBuffer.allocate(mat0.width() * mat0.height() * mat0.channels()),
            FloatBuffer.allocate(mat1.width() * mat1.height() * mat1.channels())
        )

        hwc2chw(mat0, mat0)
        hwc2chw(mat1, mat1)

        mat0.get(0, 0, fb0.array())
        mat1.get(0, 0, fb1.array())

        return mapOf(
            inputName[0] to OnnxTensor.createTensor(env, fb0, inputShape[0]),
            inputName[1] to OnnxTensor.createTensor(env, fb1, inputShape[1]),
        )
    }

    private fun processInput(mat: Mat): Mat {
        val normalized = Mat()
        mat.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255)
        return normalized
    }

    private fun cvtColor(image: Mat) {
        when (image.type()) {
            CvType.CV_8UC3 -> {}
            else -> Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2RGB)
        }
    }

    override fun load(vararg paths: String) {
        require(paths.size == 1)
        if (load) {
            session.close()
            load = false
        }
        try {
            session = env.createSession(paths[0])
        } catch (e: OrtException) {
            log.error("Could not load model, Because of ${e.message}")
        }
        initModel()
        load = true
    }

    override fun close() {
        if (load) {
            session.close()
        }
        env.close()
        load = false
    }

    override fun detect(vararg input: String): Any {
        require(input.size == 2) {
            "The input argument must be of exactly two inputs"
        }
        val (mat0, mat1) = listOf(
            Imgcodecs.imread(input[0]),
            Imgcodecs.imread(input[1]),
        )
        if (mat0.empty() || mat1.empty()) {
            log.error("Can't open image")
            throw IOException("Can't open image")
        }
        return detect(mat0, mat1)
    }

    override fun detect(vararg input: Mat): Any {
        if (!load) {
            log.error("The model must be loaded before use.")
            throw UnloadedModelException("The model must be loaded before use.")
        }
        require(input.size == 2)
        val (mat0, mat1) = input
        return detectImage(mat0, mat1)
    }
}

/*
fun main(vararg args: String) {
    val modelPath = "d:/Users/17186/Downloads/Compressed/models/text-select/siamese-model.onnx"
    val (img0, img1, img2, img3, img4) = listOf(
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/datasets/images_background/character01/1_char_1.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/datasets/images_background/character01/70_target_6.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/datasets/images_background/character02/85_target_5.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/datasets/images_background/character04/55_char_0.png",
        "C:/Users/17186/PycharmProjects/Siamese-pytorch/datasets/images_background/character29/91_target_6.png"
    )

    val modelFile = File(modelPath)
    if (!modelFile.exists()) {
        println("Model file does not exist!")
        exitProcess(-1)
    }
    val modelBytes = modelFile.readBytes()
    val model = ONNX.createSiamese(modelPath)

    val out = model.detect(img0, img4)
    println(out)
}
 */