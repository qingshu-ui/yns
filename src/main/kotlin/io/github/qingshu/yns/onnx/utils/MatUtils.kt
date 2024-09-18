package io.github.qingshu.yns.onnx.utils

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object MatUtils {

    /**
     * 矩阵转置 高，宽，通道 -> 通道，高，宽
     * @param src 原始矩阵
     * @param dst 转换后的矩阵
     */
    fun hwc2chw(src: Mat, dst: Mat) {
        val srcH = src.rows()
        val srcW = src.cols()
        val srcC = src.channels()

        src.reshape(1, srcH * srcW).run {
            Core.transpose(this, dst)
            this.release()
        }
        dst.create(srcC, srcH * srcW, src.type())
    }

    /**
     * 等比例缩放
     * @param image 图像矩阵
     * @param targetSize 缩放大小
     * @return [Mat] 新的矩阵
     */
    fun letterbox(image: Mat, targetSize: Size): Mat {
        val (w, h) = targetSize.width.toInt() to targetSize.height.toInt()
        val (iw, ih) = image.width() to image.height()
        val newImage = Mat()
        val dst = Mat(targetSize, CvType.CV_8UC3, Scalar(128.0, 128.0, 128.0))
        try {
            val scale = minOf(w.toDouble() / iw, h.toDouble() / ih)
            val (nw, nh) = (iw * scale).toInt() to (ih * scale).toInt()
            Imgproc.resize(image, newImage, Size(nw.toDouble(), nh.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
            val (xOffset, yOffset) = (w - nw) / 2 to (h - nh) / 2
            newImage.copyTo(dst.submat(Rect(xOffset, yOffset, nw, nh)))
        }finally {
            newImage.release()
        }
        return dst
    }

    /**
     * 以填充的形式调整图像，以符合目标大小
     * @param mat [Mat]
     * @param targetSize [Size]
     * @return [Mat]
     */
    @Deprecated(
        message = "There have been better implementations",
        replaceWith = ReplaceWith("MatUtils.letterbox(mat, targetSize)"),
    )
    fun scaleByPadding(mat: Mat, targetSize: Size): Mat {

        // 计算目标之间的缩放比列
        val aspectRatio = minOf(targetSize.width / mat.width(), targetSize.height / mat.height())
        // 计算缩放后的图像尺寸
        val newSize = Size(mat.width() * aspectRatio, mat.height() * aspectRatio)

        // 创建一个目标图像，并初始化为0（黑色），其尺寸为目标大小
        val dst = Mat.zeros(targetSize.height.toInt(), targetSize.width.toInt(), mat.type())
        dst.setTo(Scalar(114.0, 114.0, 114.0))

        // 创建一个用于缩放后图像的 Mat 对象
        val scaledImage = Mat()
        try {
            // 缩放源图像
            Imgproc.resize(mat, scaledImage, newSize)

            // 计算缩放后图像在目标图像中的偏移量，以使其居中对齐
            val xOffset = ((targetSize.width - newSize.width) / 2).toInt()
            val yOffset = ((targetSize.height - newSize.height) / 2).toInt()

            // 将缩放后的图像复制到目标图像的中心区域
            scaledImage.copyTo(
                dst.rowRange(yOffset, yOffset + newSize.height.toInt()).colRange(xOffset, xOffset + newSize.width.toInt())
            )
        }finally {
            scaledImage.release()
        }

        // 返回结果
        return dst
    }

    /**
     * 将 宽、高、通道 转换为 通道、宽、高
     * whc to cwh
     * @param arr [FloatArray]
     * @return [FloatArray]
     */
    fun whc2cwh(arr: FloatArray): FloatArray {
        val temp = FloatArray(arr.size)
        var j = 0
        for (ch in 0 until 3) {
            for (i in ch until arr.size step 3) {
                temp[j] = arr[i]
                j++
            }
        }
        return temp
    }

    /**
     * 矩阵转置
     * @param matrix [Array]
     * @return [Array]
     */
    fun transposeMatrix(matrix: Array<FloatArray>): Array<FloatArray> {
        val transMatrix = Array(matrix[0].size) { FloatArray(matrix.size) }
        for (i in matrix.indices) {
            for (j in matrix[0].indices) {
                transMatrix[j][i] = matrix[i][j]
            }
        }
        return transMatrix
    }

    /**
     * 获取数组中最大值的索引
     * @param arr [FloatArray]
     * @return [Int]
     */
    fun maxIndex(arr: FloatArray): Int {
        var maxVal = Float.NEGATIVE_INFINITY
        var idx = 0
        for (i in arr.indices) {
            if (arr[i] > maxVal) {
                maxVal = arr[i]
                idx = i
            }
        }
        return idx
    }

    /**
     * 调整边界框（bounding box）从原始图像尺寸到目标图像尺寸。
     *
     * 该函数将一个边界框的坐标和尺寸从原始图像尺寸（`originalSize`）映射到目标图像尺寸（`targetSize`），
     * 并保持边界框在新尺寸图像中的正确位置和比例。此操作通常用于图像处理或计算机视觉任务中，
     * 例如当图像被缩放时需要调整边界框的位置和大小以适应新的图像尺寸。
     *
     * @param bbox 边界框的数组，格式为 [x, y, width, height]，其中 x 和 y 是边界框左上角的坐标，width 和 height 是边界框的宽度和高度。
     * @param originalSize 原始图像的尺寸，包含宽度和高度。
     * @param targetSize 目标图像的尺寸，包含宽度和高度。
     */
    fun rescaleByPadding(bbox: FloatArray, originalSize: Size, targetSize: Size) {
        // 计算目标尺寸和原始尺寸之间的缩放比例
        val aspectRatio = minOf(targetSize.width / originalSize.width, targetSize.height / originalSize.height)
        // 计算调整后的图像新尺寸
        val newSize = Size(originalSize.width * aspectRatio, originalSize.height * aspectRatio)

        // 计算将图像居中对齐到目标尺寸所需的水平和垂直偏移量
        val xOffset = ((targetSize.width - newSize.width) / 2).toInt()
        val yOffset = ((targetSize.height - newSize.height) / 2).toInt()

        // 调整边界框的左上角坐标，缩放回原始尺寸下的位置
        bbox[0] = ((bbox[0] - xOffset) / aspectRatio).toFloat()
        bbox[1] = ((bbox[1] - yOffset) / aspectRatio).toFloat()

        // 调整边界框的宽度和高度
        bbox[2] = (bbox[2] / aspectRatio).toFloat()
        bbox[3] = (bbox[3] / aspectRatio).toFloat()
    }

    /**
     * 转换边界框为 xy xy
     * @param bbox [FloatArray]
     */
    fun xywh2xyxy(bbox: FloatArray) {
        val x = bbox[0]
        val y = bbox[1]
        val w = bbox[2]
        val h = bbox[3]
        bbox[0] = x - w * 0.5f
        bbox[1] = y - h * 0.5f
        bbox[2] = x + w * 0.5f
        bbox[3] = y + h * 0.5f
    }

    /**
     * 非极大值抑制
     * @param bboxList [ArrayList]
     * @param iou [Float] 交并比（Intersection over Union, IoU）
     * @return [ArrayList]
     */
    fun nonMaxSuppression(bboxList: ArrayList<FloatArray>, iou: Float): ArrayList<FloatArray> {
        val bestBoxes = ArrayList<FloatArray>()
        bboxList.sortWith(compareBy { it[4] })
        while (bboxList.isNotEmpty()) {
            val bestBox = bboxList.removeAt(bboxList.size - 1)
            bestBoxes.add(bestBox)
            bboxList.removeAll { computeIou(it, bestBox) >= iou }
        }
        return bestBoxes
    }

    /**
     * 在目标检测任务中，IoU用来衡量预测边界框与真实边界框之间的重叠程度。
     * 高的IoU意味着预测框与真实框的重叠部分更大，从而可以更准确地定位对象。
     */
    private fun computeIou(box1: FloatArray, box2: FloatArray): Float {
        // 计算第一个边界框的面积: (x_max - x_min) * (y_max - y_min)
        val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        // 计算第二个边界框的面积: (x_max - x_min) * (y_max - y_min)
        val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        // 计算相交区域的左边界: 取两个边界框左边界中的最大值
        val left = maxOf(box1[0], box2[0])
        // 计算相交区域的上边界: 取两个边界框上边界中的最大值
        val top = maxOf(box1[1], box2[1])
        // 计算相交区域的右边界: 取两个边界框右边界中的最小值
        val right = minOf(box1[2], box2[2])
        // 计算相交区域的下边界: 取两个边界框下边界中的最小值
        val bottom = minOf(box1[3], box2[3])

        // 计算相交区域的面积: 如果相交区域不存在，面积为0
        // maxOf(right - left, 0f) 和 maxOf(bottom - top, 0f) 确保面积不为负
        val interArea = maxOf(right - left, 0f) * maxOf(bottom - top, 0f)
        // 计算两个边界框的并集面积: area1 + area2 - interArea
        val unionArea = area1 + area2 - interArea

        // 计算交并比 (IoU): 相交面积除以并集面积
        // 如果 unionArea 为0，返回一个非常小的数 (1e-8f) 来避免除以零
        return maxOf(interArea / unionArea, 1e-8f)
    }
}