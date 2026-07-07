//! HTTP 客户端封装
//!
//! 基于 `reqwest` 封装，提供连接池、统一错误处理等功能。
//! 对应 Python v1 (agn-sdk) 的 `agn/core/http_client.py`（替代 httpx）。
//!
//! 设计要点：
//! - 默认启用 HTTP/2（reqwest `http2` feature）
//! - 连接池：`pool_max_idle_per_host` 控制每主机最大空闲连接
//! - 超时：总超时 + 连接超时
//! - 统一响应错误映射：`>=400` 通过 `AibridgeError::from_http_status`

use std::sync::Arc;
use std::time::Duration;

use reqwest::{Client, Method, Response};

use crate::config::ClientOptions;
use crate::error::{AibridgeError, Result};

/// 异步 HTTP 客户端
///
/// 封装 `reqwest::Client`，提供：
/// - 连接池复用
/// - 统一错误处理
/// - 请求/响应日志（tracing）
///
/// 对应 Python v1 `AsyncHttpClient`。
#[derive(Clone)]
pub struct HttpClient {
    client: Client,
    base_url: Option<String>,
}

impl HttpClient {
    /// 构建一个 HTTP 客户端
    ///
    /// `opts.base_url` 作为请求 URL 前缀（相对路径会拼接到此）。
    /// `opts.timeout` 为请求总超时；连接超时固定 30 秒。
    ///
    /// 注意：reqwest 0.12 的 `ClientBuilder` 不直接支持 base_url，
    /// 这里通过 `resolve_url` 在每次请求时手动拼接。
    pub fn new(opts: &ClientOptions) -> Result<Self> {
        let connect_timeout = Duration::from_secs(30);
        let timeout = opts.timeout_duration();

        let builder = Client::builder()
            .timeout(timeout)
            .connect_timeout(connect_timeout)
            .pool_max_idle_per_host(20)
            .https_only(false);

        let client = builder.build().map_err(AibridgeError::from)?;
        Ok(Self {
            client,
            base_url: opts.base_url.clone(),
        })
    }

    /// 用显式的 reqwest::Client 构造（测试用）
    #[cfg(test)]
    pub fn from_client(client: Client, base_url: Option<String>) -> Self {
        Self { client, base_url }
    }

    /// 返回基础 URL（如有）
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// 解析 URL：相对路径拼接 base_url，绝对路径直接使用
    fn resolve_url(&self, url: &str) -> String {
        if url.starts_with("http://") || url.starts_with("https://") {
            return url.to_string();
        }
        match &self.base_url {
            Some(base) => {
                let base = base.trim_end_matches('/');
                let path = url.trim_start_matches('/');
                format!("{base}/{path}")
            }
            None => url.to_string(),
        }
    }

    /// 发送 GET 请求
    pub async fn get(&self, url: &str) -> Result<Response> {
        self.request(Method::GET, url).await
    }

    /// 发送 POST 请求（JSON body）
    pub async fn post_json<T: serde::Serialize + ?Sized>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<Response> {
        let resp = self
            .client
            .post(self.resolve_url(url))
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        handle_response(resp).await
    }

    /// 发送 POST 请求（原始字节 body）
    pub async fn post_bytes(
        &self,
        url: &str,
        content_type: &str,
        body: bytes::Bytes,
    ) -> Result<Response> {
        let resp = self
            .client
            .post(self.resolve_url(url))
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        handle_response(resp).await
    }

    /// 发送带自定义请求构造的请求（适配器需要自定义 header/body 时用）
    pub async fn request(&self, method: Method, url: &str) -> Result<Response> {
        let resp = self
            .client
            .request(method, self.resolve_url(url))
            .send()
            .await
            .map_err(map_reqwest_error)?;
        handle_response(resp).await
    }

    /// 发送带 Bearer 认证的 JSON 请求
    pub async fn post_json_authed<T: serde::Serialize + ?Sized>(
        &self,
        url: &str,
        api_key: &str,
        body: &T,
    ) -> Result<Response> {
        let resp = self
            .client
            .post(self.resolve_url(url))
            .bearer_auth(api_key)
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        handle_response(resp).await
    }

    /// 发送带 Bearer 认证的 GET 请求
    pub async fn get_authed(&self, url: &str, api_key: &str) -> Result<Response> {
        let resp = self
            .client
            .get(self.resolve_url(url))
            .bearer_auth(api_key)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        handle_response(resp).await
    }

    /// 获取底层 reqwest::Client 的引用（适配器需要更细粒度控制时用）
    pub fn inner(&self) -> &Client {
        &self.client
    }
}

/// 将 reqwest::Error 映射为 AibridgeError
///
/// 超时 → Timeout；其余 → Network。
fn map_reqwest_error(err: reqwest::Error) -> AibridgeError {
    if err.is_timeout() {
        AibridgeError::Timeout
    } else {
        AibridgeError::Network(err)
    }
}

/// 处理响应：状态码 >= 400 转为错误
///
/// 对应 Python v1 `_handle_response`。
async fn handle_response(resp: Response) -> Result<Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let status_code = status.as_u16();
    let body_text = resp.text().await.unwrap_or_default();
    Err(AibridgeError::from_http_status(status_code, &body_text))
}

/// 共享的 HTTP 客户端句柄
pub type SharedHttpClient = Arc<HttpClient>;

/// 将 HttpClient 转为共享句柄
pub fn shared(client: HttpClient) -> SharedHttpClient {
    Arc::new(client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientOptions;

    fn make_client() -> HttpClient {
        HttpClient::new(&ClientOptions::default()).expect("构建客户端失败")
    }

    #[test]
    fn resolve_url_absolute_passthrough() {
        let c = make_client();
        assert_eq!(
            c.resolve_url("https://api.example.com/v1"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            c.resolve_url("http://localhost:8080"),
            "http://localhost:8080"
        );
    }

    #[test]
    fn resolve_url_relative_joins_base() {
        let opts = ClientOptions::builder()
            .base_url("https://api.example.com/")
            .build();
        let c = HttpClient::new(&opts).unwrap();
        assert_eq!(c.resolve_url("/v1/chat"), "https://api.example.com/v1/chat");
        assert_eq!(c.resolve_url("v1/chat"), "https://api.example.com/v1/chat");
    }

    #[test]
    fn resolve_url_no_base_returns_as_is() {
        let c = make_client();
        assert_eq!(c.resolve_url("/v1/chat"), "/v1/chat");
    }

    #[test]
    fn base_url_exposed() {
        let opts = ClientOptions::builder()
            .base_url("https://api.example.com")
            .build();
        let c = HttpClient::new(&opts).unwrap();
        assert_eq!(c.base_url(), Some("https://api.example.com"));
    }

    #[test]
    fn client_constructs_with_pool_and_timeout() {
        let opts = ClientOptions::builder().timeout(42).build();
        let c = HttpClient::new(&opts).unwrap();
        // 内部 client 应存在且可用（无法直接断言 timeout，但能拿到 inner 即可）
        let _inner = c.inner();
    }

    #[test]
    fn client_with_invalid_base_url_still_constructs() {
        // base_url 仅作字符串保留（手动拼接），任何字符串都不会导致构造失败
        let opts = ClientOptions::builder()
            .base_url("not a url at all")
            .build();
        let c = HttpClient::new(&opts);
        assert!(c.is_ok());
        if let Ok(c) = c {
            assert_eq!(c.base_url(), Some("not a url at all"));
        }
    }

    #[test]
    fn from_http_status_maps_401() {
        let err = AibridgeError::from_http_status(401, "");
        assert!(matches!(err, AibridgeError::Authentication { .. }));
    }

    #[test]
    fn shared_client_wraps_in_arc() {
        let c = make_client();
        let s = shared(c);
        // Arc 引用计数为 1
        assert_eq!(Arc::strong_count(&s), 1);
    }
}
