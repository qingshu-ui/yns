package io.github.qingshu.yns.aop

import io.github.qingshu.yns.annotation.MeasureTime
import io.github.qingshu.yns.annotation.Slf4j
import io.github.qingshu.yns.annotation.Slf4j.Companion.log
import org.aspectj.lang.ProceedingJoinPoint
import org.aspectj.lang.annotation.Around
import org.aspectj.lang.annotation.Aspect
import org.springframework.stereotype.Component
import kotlin.system.measureTimeMillis

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the AGPL-3.0 License.
 * See the LICENSE file for details.
 */
@Slf4j
@Aspect
@Component
class MeasureTimeAspect {

    @Around("@annotation(measure)")
    fun measureTime(joinPoint: ProceedingJoinPoint, measure: MeasureTime): Any? {
        var result = Any()
        return try {
            val executedTime = measureTimeMillis {
                result = joinPoint.proceed()
            }
            val resultClass = result::class.java
            val timeField = resultClass.getDeclaredField(measure.fieldName)
            timeField.isAccessible = true
            timeField.set(result, executedTime)
            result
        } catch (e: Exception) {
            log.error(e.message)
            result
        }
    }
}