package io.github.qingshu.yns.dto

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
}
