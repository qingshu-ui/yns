package io.github.qingshu.yns.dto

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
data class Detection(
    /**
     * 标签
     */
    var label: String,

    /**
     * 标签索引
     */
    var labelIndex:Int,

    /**
     * 边界框值
     */
    var bbox: FloatArray,

    /**
     * 置信度，可信度
     */
    var confidence:Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Detection

        if (label != other.label) return false
        if (labelIndex != other.labelIndex) return false
        if (!bbox.contentEquals(other.bbox)) return false
        if (confidence != other.confidence) return false

        return true
    }

    override fun hashCode(): Int {
        var result = label.hashCode()
        result = 31 * result + labelIndex
        result = 31 * result + bbox.contentHashCode()
        result = 31 * result + confidence.hashCode()
        return result
    }

    @Deprecated(
        message = "This method will be removed in a future version",
    )
    fun drawWithIndex(img: Mat, index: Int) {
        val color = Scalar(220.0, 50.0, 0.0)
        Imgproc.rectangle(
            img,
            Point(bbox[0].toDouble(), bbox[1].toDouble()), Point(bbox[2].toDouble(), bbox[3].toDouble()),
            color,
            2,
            )
        Imgproc.putText(
            img,
            "$index",
            Point(bbox[0] - 1.0, bbox[1] - 5.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    }
}
