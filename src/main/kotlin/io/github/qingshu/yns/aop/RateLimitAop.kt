package io.github.qingshu.yns.aop

import io.github.qingshu.yns.annotation.RateLimit
import io.github.qingshu.yns.annotation.Slf4j
import org.aspectj.lang.ProceedingJoinPoint
import org.aspectj.lang.annotation.Around
import org.aspectj.lang.annotation.Aspect
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.stereotype.Component
import org.springframework.web.context.request.RequestContextHolder
import org.springframework.web.context.request.ServletRequestAttributes
import java.util.concurrent.ConcurrentHashMap

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
class RateLimitAop {

    private val requestCounts = ConcurrentHashMap<String, MutableList<Long>>()

    @Around("@annotation(limiter)")
    fun rateLimit(joinPoint: ProceedingJoinPoint, limiter: RateLimit): Any {
        val request = (RequestContextHolder.currentRequestAttributes() as ServletRequestAttributes).request
        val clientIp = request.remoteAddr
        val now = System.currentTimeMillis()

        requestCounts.putIfAbsent(clientIp, mutableListOf())
        val timestamps = requestCounts[clientIp]!!

        timestamps.removeIf { it < now - limiter.timeWindow }

        if (timestamps.size >= limiter.limit) {
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                .body(mapOf("error" to "Too many requests, please try again later"))
        }

        timestamps.add(now)
        return joinPoint.proceed()
    }
}