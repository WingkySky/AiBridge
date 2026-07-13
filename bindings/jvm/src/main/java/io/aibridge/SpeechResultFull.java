package io.aibridge;

/**
 * 文字转语音完整结果（meta + 二进制音频）。
 *
 * <p>Rust 侧 {@code SpeechResult.audio_data} 不参与 serde，二进制通过 FFI 的
 * {@code aibridge_bytes_t} 单独传递。本类把 meta JSON 与二进制音频合并为单一返回，
 * 便于调用方使用。
 */
public class SpeechResultFull {

    /** meta 信息（content_type/format/duration/model 等） */
    private final SpeechResult meta;
    /** 二进制音频数据（若 Provider 返回二进制；否则为空数组） */
    private final byte[] audioData;

    public SpeechResultFull(SpeechResult meta, byte[] audioData) {
        this.meta = meta;
        this.audioData = audioData == null ? new byte[0] : audioData;
    }

    /** meta 信息 */
    public SpeechResult getMeta() {
        return meta;
    }

    /** 二进制音频数据 */
    public byte[] getAudioData() {
        return audioData;
    }

    /** 音频长度（便捷方法） */
    public int audioLength() {
        return audioData.length;
    }
}
