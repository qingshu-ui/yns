package io.github.qingshu.yns.dto

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
enum class DetectStatus {
    /**
     * 初始化
     */
    INITIALIZE,

    /**
     * 数据准备
     */
    DATA_PREPARATION,

    /**
     * 前处理
     */
    PREPROCESS,

    /**
     * 推理
     */
    INFERENCE,

    /**
     * 后处理
     */
    POSTPROCESS,

    /**
     * 完成
     */
    COMPLETED,

    /**
     * 错误
     */
    ERROR,
}