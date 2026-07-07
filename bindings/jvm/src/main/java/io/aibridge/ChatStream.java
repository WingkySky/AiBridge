package io.aibridge;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.lang.ref.Cleaner;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 流式文本对话句柄（封装 native stream 句柄）。
 *
 * <p>由 {@link Client#chatStream} 创建。提供两种消费方式：
 * <ol>
 *   <li>{@link Iterator}（阻塞遍历）：{@code for (chunk : stream) { ... }}</li>
 *   <li>{@link Flow.Publisher}（反应式）：{@link #subscribe}</li>
 * </ol>
 *
 * <h3>FFI 遗留约束</h3>
 * <ul>
 *   <li><b>stream_next 串行</b>：同一 stream 不可并发 next。{@link #hasNext} /
 *       {@link #next} 用 synchronized 保证串行；反应式订阅在单线程串行拉取。</li>
 *   <li><b>chunk JSON 必须释放</b>：每个 chunk 的 {@code out_chunk_json} 由 Rust 分配，
 *       反序列化后立即 {@code aibridge_string_free}。</li>
 *   <li><b>stream 句柄必须 destroy</b>：用 {@link Cleaner} 兜底，建议显式 close
 *       （try-with-resources）。destroy 触发 Rust drop → tokio task abort，实现取消语义。</li>
 * </ul>
 *
 * <h3>错误传递</h3>
 * <p>流式过程中 {@code stream_next} 返回负数时，<b>同线程立即</b>读取
 * {@code aibridge_last_error()} 转存为异常。Iterator 模式下抛出；反应式模式下通过
 * {@link java.util.concurrent.Flow.Subscriber#onError} 传递。
 */
public class ChatStream implements Iterator<ChatCompletionChunk>, Flow.Publisher<ChatCompletionChunk>, AutoCloseable {

    private static final Cleaner CLEANER = Cleaner.create();
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

    /** native stream 句柄（null 表示已结束/关闭） */
    private volatile Pointer handle;
    /** 防止重复 close */
    private final AtomicBoolean closed = new AtomicBoolean(false);
    /** Cleaner 兜底释放 */
    private final Cleaner.Cleanable cleanable;

    /** 预取的下一个 chunk（null 表示未预取或已结束） */
    private ChatCompletionChunk nextChunk;
    /** 是否已到达 EOF */
    private boolean eof = false;
    /** 流式错误（EOF 时若非 null 则表示以错误结束） */
    private AibridgeException error;

    public ChatStream(Pointer streamHandle) {
        if (streamHandle == null) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "stream 句柄为空", null, false);
        }
        this.handle = streamHandle;
        this.cleanable = CLEANER.register(this, () -> AibridgeNative.INSTANCE.aibridge_stream_destroy(streamHandle));
    }

    // —— Iterator 模式（阻塞，串行）—— //

    @Override
    public synchronized boolean hasNext() {
        if (eof) {
            return false;
        }
        if (nextChunk != null) {
            return true;
        }
        // 预取下一个 chunk
        pullNext();
        return nextChunk != null;
    }

    @Override
    public synchronized ChatCompletionChunk next() {
        if (eof && nextChunk == null) {
            if (error != null) {
                throw error;
            }
            throw new NoSuchElementException("流已结束");
        }
        if (nextChunk == null) {
            pullNext();
            if (nextChunk == null) {
                if (error != null) {
                    throw error;
                }
                throw new NoSuchElementException("流已结束");
            }
        }
        ChatCompletionChunk chunk = nextChunk;
        nextChunk = null;
        return chunk;
    }

    /** 取流式错误（若以错误结束）；正常结束返回 null。须在遍历结束后调用。 */
    public AibridgeException getError() {
        return error;
    }

    // —— Flow.Publisher 模式（反应式）—— //

    /**
     * 反应式订阅：在单线程串行拉取 chunk 并推送给 subscriber。
     *
     * <p>背压：subscriber 通过 {@code Subscription.request(n)} 申请 chunk，本实现用
     * {@link java.util.concurrent.Semaphore} 计数控制，每次 request 释放许可，拉取循环
     * 获取许可后才推下一个 chunk。取消（{@code cancel}）后停止拉取。
     *
     * <p>FFI 串行约束：所有 {@code stream_next} 在单一拉取线程调用，天然串行。
     */
    @Override
    public void subscribe(Flow.Subscriber<? super ChatCompletionChunk> subscriber) {
        if (subscriber == null) {
            throw new NullPointerException("subscriber 不能为空");
        }
        final java.util.concurrent.Semaphore permits = new java.util.concurrent.Semaphore(0);
        final java.util.concurrent.atomic.AtomicBoolean cancelled = new java.util.concurrent.atomic.AtomicBoolean(false);

        Flow.Subscription subscription = new Flow.Subscription() {
            @Override
            public void request(long n) {
                if (n <= 0) {
                    subscriber.onError(new IllegalArgumentException("request 需正数: " + n));
                    return;
                }
                permits.release((int) Math.min(n, Integer.MAX_VALUE));
            }

            @Override
            public void cancel() {
                cancelled.set(true);
                permits.release(Integer.MAX_VALUE);
            }
        };

        subscriber.onSubscribe(subscription);

        // 单线程串行拉取（保证 stream_next 串行，满足 FFI 约束）
        Thread.ofVirtual().name("aibridge-stream-pull").start(() -> {
            try {
                while (!cancelled.get()) {
                    permits.acquire();
                    if (cancelled.get()) {
                        break;
                    }
                    if (!hasNext()) {
                        // 流结束
                        if (error != null) {
                            subscriber.onError(error);
                        } else {
                            subscriber.onComplete();
                        }
                        return;
                    }
                    subscriber.onNext(next());
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                subscriber.onError(e);
            } catch (Exception e) {
                subscriber.onError(e);
            }
        });
    }

    // —— 生命周期 —— //

    /** 关闭流，释放 stream 句柄。多次调用安全。 */
    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        eof = true;
        cleanable.clean();
        handle = null;
    }

    // —— 内部：串行拉取下一个 chunk —— //

    /**
     * 调用 {@code aibridge_stream_next} 拉取并解析下一个 chunk。
     *
     * <p>同步块保证串行（FFI 遗留：同一 stream 不可并发 next）。
     * 结果写入 {@link #nextChunk}；EOF 时设 {@link #eof}；错误时设 {@link #error}。
     */
    private synchronized void pullNext() {
        if (eof || handle == null) {
            return;
        }
        PointerByReference outRef = new PointerByReference();
        int status = AibridgeNative.INSTANCE.aibridge_stream_next(handle, outRef);

        if (status == AibridgeNative.AIBRIDGE_STREAM_CHUNK) {
            Pointer jsonPtr = outRef.getValue();
            if (jsonPtr == null) {
                error = new AibridgeException(AibridgeException.CODE_FFI,
                        "stream_next 返回 chunk 但 out_chunk_json 为空", null, false);
                eof = true;
                return;
            }
            try {
                String json = jsonPtr.getString(0, "UTF-8");
                nextChunk = parseChunk(json);
            } finally {
                // chunk JSON 由 Rust 分配，必须释放
                AibridgeNative.INSTANCE.aibridge_string_free(jsonPtr);
            }
        } else if (status == AibridgeNative.AIBRIDGE_STREAM_END) {
            eof = true;
        } else {
            // 负数：错误，立即同线程读取 last_error 转存
            error = readLastError();
            eof = true;
        }
    }

    /** 解析 chunk JSON */
    private static ChatCompletionChunk parseChunk(String json) {
        try {
            return MAPPER.readValue(json, ChatCompletionChunk.class);
        } catch (Exception e) {
            throw new AibridgeException(AibridgeException.CODE_FFI,
                    "ChatCompletionChunk JSON 反序列化失败: " + e.getMessage()
                            + " (原始: " + json + ")", null, false, e);
        }
    }

    /** 读取 last_error 并映射为异常（同线程立即读取） */
    private static AibridgeException readLastError() {
        Pointer errPtr = AibridgeNative.INSTANCE.aibridge_last_error();
        if (errPtr == null) {
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "未知错误（last_error 为空）", null, false);
        }
        String json = errPtr.getString(0, "UTF-8");
        try {
            ErrorPayload payload = MAPPER.readValue(json, ErrorPayload.class);
            String code = payload.code != null ? payload.code : AibridgeException.CODE_FFI;
            String details = payload.details != null ? payload.details : "null";
            boolean retryable = Boolean.TRUE.equals(payload.retryable);
            String message = payload.message != null ? payload.message : "(无错误消息)";
            return new AibridgeException(code, message, details, retryable);
        } catch (Exception e) {
            return new AibridgeException(AibridgeException.CODE_FFI,
                    "last_error JSON 解析失败: " + e.getMessage(), null, false, e);
        }
    }

    private static class ErrorPayload {
        public String code;
        public String message;
        public String details;
        public Boolean retryable;
    }
}
