package io.github.qingshu.yns.annotation

import org.slf4j.LoggerFactory

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.RUNTIME)
annotation class Slf4j {
    companion object {
        val <reified T> T.log: org.slf4j.Logger
            inline get() = LoggerFactory.getLogger(T::class.java)
    }
}