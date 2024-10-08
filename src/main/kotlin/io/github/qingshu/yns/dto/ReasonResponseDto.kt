package io.github.qingshu.yns.dto

import com.fasterxml.jackson.annotation.JsonProperty

data class ReasonResponseDto(
    @JsonProperty("inference-time")
    var reasonTime: Long,
    @JsonProperty("image-url")
    val imageUrl: String,
    val detections: List<Detection>,
)
