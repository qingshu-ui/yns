package io.github.qingshu.yns.service

import io.github.qingshu.yns.dto.Detection
import io.github.qingshu.yns.onnx.impl.SiameseOnnxModel
import io.github.qingshu.yns.onnx.impl.YoloOnnxModel
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import java.nio.file.Files
import kotlin.io.path.Path

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
class TextSelectCaptchaImpl(
    private val yoloModel: YoloOnnxModel,
    private val siameseModel: SiameseOnnxModel,
    private val labelPath: String,
): TextSelectCaptcha, AutoCloseable {
    private val labels = Files.lines(Path(labelPath)).toList()

    override fun run(imagePath: String): List<Detection> {
        val inputMat = Imgcodecs.imread(imagePath)
        require(!inputMat.empty())
        return process(inputMat)
    }

    override fun run(image: Mat): List<Detection> {
        require(!image.empty())
        return process(image)
    }

    override fun close() {
        yoloModel.close()
        siameseModel.close()
    }

    private fun process(image: Mat): List<Detection> {
        val result = yoloModel.detect(image, labels)

        val chars = result.filter { it.label == labels[0] }.toMutableList()
        val targets = result.filter { it.label == labels[1] }.toMutableList()
        require(chars.size == targets.size)

        chars.sortBy { it.bbox[0] }

        val charMats = chars.map {
            val (xMin, yMin, xMax, yMax) = it.bbox
            image.submat(Rect(xMin.toInt(), yMin.toInt(), (xMax - xMin).toInt(), (yMax - yMin).toInt()))
        }

        val matchedTargets = mutableListOf<Detection>()

        for(charMat in charMats) {
            var bestMatchIndex = -1
            var bestSimilarity = Float.NEGATIVE_INFINITY

            for((targetIndex, target) in targets.withIndex()) {
                val (xMin, yMin, xMax, yMax) = target.bbox
                val targetMat = image.submat(
                    Rect(xMin.toInt(), yMin.toInt(), (xMax - xMin).toInt(), (yMax - yMin).toInt())
                )
                val similarity = siameseModel.detect(charMat, targetMat)

                if(similarity > bestSimilarity) {
                    bestSimilarity = similarity
                    bestMatchIndex = targetIndex
                }
            }
            matchedTargets.add(targets[bestMatchIndex])
            targets.removeAt(bestMatchIndex)
        }
        return matchedTargets
    }

    @Deprecated(
        message = "This method will be removed in a future version.",
    )
    fun showImage(img: Mat, windowName: String = "showImage"){
        HighGui.imshow(windowName, img)
        HighGui.waitKey(0)
        HighGui.destroyWindow(windowName)
    }
}

/* Example
fun main(vararg args: String) {
    OpenCV.loadLocally()
    val imagePath = "C:\\Users\\17186\\Desktop\\labelme\\siamese\\siamese-datasets\\0.png"
    val (yoloModelPath, yoloLabelPath, siameseModelPath) = listOf(
        "d:\\Users\\17186\\Downloads\\Compressed\\models\\text-select\\text-select-yolov8s.onnx",
        "d:\\Users\\17186\\Downloads\\Compressed\\models\\text-select\\text-select-label.names",
        "d:\\Users\\17186\\Downloads\\Compressed\\models\\text-select\\siamese_model.onnx"
    )
    val yoloModel = YoloOnnxModel(yoloModelPath)
    val siameseModel = SiameseOnnxModel(siameseModelPath)
    val image = Imgcodecs.imread(imagePath)
    val textSelectCaptchaImpl =
        TextSelectCaptchaImpl(yoloModel = yoloModel, siameseModel = siameseModel, labelPath = yoloLabelPath)
    val result = textSelectCaptchaImpl.run(image)
    result.forEachIndexed { index, detection ->
        detection.drawWithIndex(image,index)
    }
    textSelectCaptchaImpl.showImage(image)
}
 */