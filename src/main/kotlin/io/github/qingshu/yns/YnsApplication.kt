package io.github.qingshu.yns

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.context.annotation.EnableAspectJAutoProxy
import org.springframework.scheduling.annotation.EnableScheduling

@EnableAspectJAutoProxy
@EnableScheduling
@SpringBootApplication
class YnsApplication

fun main(args: Array<String>) {
    runApplication<YnsApplication>(*args)
}
