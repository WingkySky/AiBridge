// AIBridge JVM 绑定构建脚本
//
// 通过 JNA 调用 aibridge-ffi 的 cdylib（libaibridge.dylib / .so / .dll）。
// 依赖：
// - net.java.dev.jna:jna：纯 Java 调 C ABI
// - com.fasterxml.jackson.*：JSON 边界序列化/反序列化
//
// 运行 hello world：
//   ./gradlew run
// 需通过 -Djava.library.path 或环境变量 DYLD_LIBRARY_PATH / LD_LIBRARY_PATH
// 指向 libaibridge 所在目录（默认 target/debug）。

plugins {
    application
    java
}

group = "io.aibridge"
version = "0.1.0"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    // JNA：纯 Java 调 C ABI，无需 native Rust 绑定
    implementation("net.java.dev.jna:jna:5.15.0")
    // Jackson：JSON 边界（反）序列化
    implementation("com.fasterxml.jackson.core:jackson-databind:2.18.1")
    // JSR305 注解（@Nullable 等），提升 JNA Pointer 语义可读性
    implementation("com.google.code.findbugs:jsr305:3.0.2")
}

application {
    // Hello world 入口
    mainClass = "io.aibridge.Hello"
    // 默认把 cargo debug 输出目录加入 java.library.path（可被命令行覆盖）
    applicationDefaultJvmArgs = listOf(
        "-Djava.library.path=${rootProject.projectDir}/../../target/debug",
        "-Djna.library.path=${rootProject.projectDir}/../../target/debug"
    )
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
    // 保留参数名（便于反射/Jackson）
    options.compilerArgs.addAll(listOf("-parameters"))
}

tasks.withType<JavaExec> {
    // 传递系统属性（如 -D...），便于调试
    systemProperties = System.getProperties() as Map<String, Any>
}

tasks.test {
    useJUnitPlatform()
}
