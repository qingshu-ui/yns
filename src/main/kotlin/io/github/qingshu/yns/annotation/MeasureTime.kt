package io.github.qingshu.yns.annotation

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the APGL-3.0 License.
 * See the LICENSE file for details.
 */
@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.FUNCTION)
annotation class MeasureTime (
    val fieldName: String = "time",
)