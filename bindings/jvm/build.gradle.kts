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

// ──────────────────────────────────────────────────────────────────────────
// 动态库打进 jar（发布分发，设计文档 12.1）
//
// copyNativeLib：把 target/release 的 libaibridge 拷到 build/resources/main/{os}/{arch}/
// （JNA classpath 约定路径，运行时 Native.load 自动从 jar 内加载）。
// 由 -PembedNative=true 触发（CI/发布用）；本地 ./gradlew run 仍走 java.library.path。
// jar task 按平台加 classifier（如 darwin-aarch64），便于多平台并行发布。
// ──────────────────────────────────────────────────────────────────────────

// 计算 native 库的 OS/arch 标识（JNA 约定：darwin/linux/win32 × x86_64/aarch64）
val nativeOs: String = when {
    System.getProperty("os.name").lowercase().contains("mac") -> "darwin"
    System.getProperty("os.name").lowercase().contains("linux") -> "linux"
    System.getProperty("os.name").lowercase().contains("windows") -> "win32"
    else -> "unknown"
}
val nativeArch: String = when (System.getProperty("os.arch").lowercase()) {
    "aarch64", "arm64" -> "aarch64"
    "x86_64", "amd64" -> "x86_64"
    else -> System.getProperty("os.arch")
}

// 动态库文件名（按平台）
val nativeLibName: String = when (nativeOs) {
    "darwin" -> "libaibridge.dylib"
    "linux" -> "libaibridge.so"
    "win32" -> "aibridge.dll"
    else -> "libaibridge.unknown"
}

// release 动态库目录（cargo build -p aibridge-ffi --release 产物）
val ffiReleaseDir = file("${rootProject.projectDir}/../../target/release")

// 拷贝 libaibridge 进 jar resources（JNA classpath 约定路径 {os}/{arch}/）
tasks.register<Copy>("copyNativeLib") {
    description = "把 libaibridge 拷进 build/resources/main/{os}/{arch}/（打进 jar）"
    group = "build"
    from(ffiReleaseDir) {
        include(nativeLibName)
    }
    into(layout.buildDirectory.dir("resources/main/$nativeOs/$nativeArch"))
    // 仅 -PembedNative=true 且 release 产物存在时执行（避免本地构建因无产物报错）
    onlyIf {
        project.hasProperty("embedNative") && file("${ffiReleaseDir}/$nativeLibName").exists()
    }
}

// jar 依赖 copyNativeLib（确保 native 打进 jar），发布时按平台加 classifier
tasks.jar {
    dependsOn("copyNativeLib")
    if (project.hasProperty("embedNative")) {
        archiveClassifier.set("$nativeOs-$nativeArch")
    }
}
