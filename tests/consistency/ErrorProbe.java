import io.aibridge.AibridgeException;
import io.aibridge.Client;

/**
 * 跨语言错误一致性探针（JVM）：未知 provider 必须返回 code == "provider_not_found"。
 *
 * <p>JVM 绑定从 FFI last_error JSON 解析 code 字段（{@link AibridgeException#getCode()}），
 * code 直接来自 core {@code AibridgeError::code()}。
 *
 * <p>编译运行（classpath 含 jvm classes + jna jar）：
 * <pre>{@code
 * javac -cp <jvm-classes>:<jna.jar> tests/consistency/ErrorProbe.java
 * java -Djna.library.path=target/debug -cp <jvm-classes>:<jna.jar>:tests/consistency ErrorProbe
 * }</pre>
 *
 * 退出码 0 表示通过，1 表示失败。
 */
public class ErrorProbe {

    private static final String EXPECTED = "provider_not_found";

    public static void main(String[] args) {
        // 未知 provider + 假 key（跳过 key 校验，触发 ProviderNotFound）
        // ClientOptions JSON 字段为 snake_case（core serde 默认）
        String configJson = "{\"api_key\":\"dummy-key\"}";
        try {
            new Client("nonexistent", configJson);
        } catch (AibridgeException e) {
            if (EXPECTED.equals(e.getCode())) {
                System.out.println("[jvm] OK：e.getCode()=\"" + EXPECTED + "\"");
                System.out.println("[jvm]   message=" + e.getMessage());
                System.exit(0);
            }
            System.out.println("[jvm] FAIL：期望 code=\"" + EXPECTED + "\"，实际 code=\"" + e.getCode() + "\"");
            System.out.println("[jvm]   message=" + e.getMessage());
            System.exit(1);
        }
        System.out.println("[jvm] FAIL：未抛出任何异常");
        System.exit(1);
    }
}
