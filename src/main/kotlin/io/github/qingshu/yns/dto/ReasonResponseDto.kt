package io.github.qingshu.yns.dto

data class ReasonResponseDto(
    val imageUrl: String,
    val detections: List<Detection>,
)
