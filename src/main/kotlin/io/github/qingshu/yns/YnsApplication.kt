package io.github.qingshu.yns

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication
import org.springframework.scheduling.annotation.EnableScheduling

@EnableScheduling
@SpringBootApplication
class YnsApplication

fun main(args: Array<String>) {
    runApplication<YnsApplication>(*args)
}
