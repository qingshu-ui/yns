package io.github.qingshu.yns.onnx.utils


import ai.onnxruntime.OrtSession
import io.github.qingshu.yns.annotation.Slf4j
import io.github.qingshu.yns.annotation.Slf4j.Companion.log
import io.github.qingshu.yns.onnx.impl.YoloOnnxModel
import nu.pattern.OpenCV
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs.imread
import org.opencv.imgcodecs.Imgcodecs.imwrite
import org.opencv.imgproc.Imgproc.resize
import java.io.File

@Slf4j
object YoloV8Utils {

    /**
     * 将检测的对象裁剪为 Siamese.kt 数据图片 105 * 105
     * @param inputPath 图片所在目录
     * @param outPath 输出目录
     * @param model YOLO 模型
     */
    fun clipImgToSiamese(inputPath: String, outPath: String, model: YoloOnnxModel, labelPath: String)  {
        val inputDir = File(inputPath).also { it.mkdirs() }
        val outputDir = File(outPath).also { it.mkdirs() }
        if (!inputDir.isDirectory || !outputDir.isDirectory) {
            log.warn("'inputPath' or 'outPath' is not a directory")
            return
        }

        inputDir.listFiles { file -> file.isFile && isImageFile(file) }?.forEach { imageFile ->
            //log.info("Processing image: ${imageFile.name}")

            // Read input image
            val originalImage = imread(imageFile.absolutePath)
            if (originalImage.empty()) {
                log.warn("Failed to load image: ${imageFile.absolutePath}")
                return@forEach
            }

            // Detecting object using YOLO
            val detections = model.detect(originalImage, labelPath = labelPath, conf = 0.51f)

            // Processing each bbox, crop it and adjust it to 105 * 105
            detections.forEachIndexed { index, (label, _, bbox) ->
                val (xMin, yMin, xMax, yMax) = bbox

                // Convert to Rect
                val rect = Rect(xMin.toInt(), yMin.toInt(), (xMax - xMin).toInt(), (yMax - yMin).toInt())

                // Clip
                val cropped = Mat(originalImage, rect)

                // Resize to 105*105
                val resized = Mat()
                resize(cropped, resized, Size(105.0, 105.0))

                // Save the processed image
                val outImagePath = "$outPath/${imageFile.nameWithoutExtension}_${label}_${index}.${imageFile.extension}"
                imwrite(outImagePath, resized)
                //log.info("Saved image to: $outImagePath")

                // release img
                cropped.release()
                resized.release()
            }
        }
        println("All image processed")
    }

    private fun isImageFile(file: File): Boolean {
        val imageExtension = listOf("jpg", "jpeg", "png", "tiff", "bmp")
        return imageExtension.any { file.extension.equals(it, ignoreCase = true) }
    }
}

fun main(args: Array<String>) {
    OpenCV.loadLocally()
    val (modelPath, labelPath, inputPath, outPath) = listOf(
        "d:/Users/17186/Downloads/Compressed/models/text-select/text-select-yolov8s.onnx",
        "d:/Users/17186/Downloads/Compressed/models/text-select/text-select-label.names",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets",
        "C:/Users/17186/Desktop/labelme/siamese/siamese-datasets/out2"
    )
    //val model = YOLO8.newInstance(modelPath = modelPath, labelPath = labelPath)
    val modelLoadStartTime = System.currentTimeMillis()
    val model = YoloOnnxModel(modelPath, OrtSession.SessionOptions())
    val modelLoadEndTime = System.currentTimeMillis()
    println("Model loaded in ${modelLoadEndTime - modelLoadStartTime} ms")
    val startTime = System.currentTimeMillis()
    model.use {
        YoloV8Utils.clipImgToSiamese(inputPath, outPath, model, labelPath)
    }
    val endTime = System.currentTimeMillis()
    println("Completed in ${endTime - startTime} ms")
}
