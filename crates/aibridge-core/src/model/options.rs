//! 统一请求选项与工具定义
//!
//! 定义所有 AI 能力的标准化请求参数及工具/函数调用相关类型。
//! 对应 Python v1 (agn-sdk) 的 `agn/models/options.py`。
//!
//! 设计原则（与设计文档第 6 节一致）：
//! - Rust 不保留 Python 的 `ChatOptions/ImageOptions` 中间层，改为 `Request::builder()` 链式调用
//! - 厂商特有参数通过 `extra: HashMap<String, Value>` 透传
//! - 工具调用、响应格式等通用类型在此统一定义

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 推理努力程度（统一思考模式）
///
/// 对应 Python v1 `ReasoningEffort`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
    Auto,
}

/// 响应格式
///
/// 对应 Python v1 `ResponseFormat`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// 纯文本
    Text,
    /// JSON 对象
    JsonObject,
    /// JSON Schema（结构化输出）
    JsonSchema,
}

/// 工具选择策略
///
/// 对应 Python v1 `ToolChoice`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// 不调用工具
    None,
    /// 自动决定
    Auto,
    /// 强制调用
    Required,
}

/// 停止词
///
/// 单个或多个停止词。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSeq {
    /// 单个停止词
    Single(String),
    /// 多个停止词
    Multiple(Vec<String>),
}

/// 函数定义
///
/// 对应 Python v1 `FunctionDefinition`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// 函数名称
    pub name: String,
    /// 函数描述
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// 函数参数（JSON Schema）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// 工具定义
///
/// 对应 Python v1 `ToolDefinition`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// 工具类型
    #[serde(default = "default_tool_type")]
    #[serde(rename = "type")]
    pub tool_type: String,
    /// 函数定义（type=function 时必填）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionDefinition>,
}

impl ToolDefinition {
    /// 创建函数类型工具
    pub fn function(func: FunctionDefinition) -> Self {
        Self {
            tool_type: "function".into(),
            function: Some(func),
        }
    }

    /// 创建 web_search 工具
    pub fn web_search() -> Self {
        Self {
            tool_type: "web_search".into(),
            function: None,
        }
    }
}

fn default_tool_type() -> String {
    "function".into()
}

/// 工具调用（模型生成的调用请求）
///
/// 对应 Python v1 `ToolCall`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// 工具调用 ID
    pub id: String,
    /// 工具类型（通常为 "function"）
    #[serde(default = "default_tool_type")]
    #[serde(rename = "type")]
    pub tool_type: String,
    /// 函数调用信息：包含 `name` 和 `arguments`（arguments 为 JSON 字符串）
    pub function: ToolCallFunction,
}

/// 工具调用中的函数部分
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// 函数名
    pub name: String,
    /// 函数参数（JSON 字符串，由模型生成）
    pub arguments: String,
}

/// 通用参数映射规则
///
/// 对应 Python v1 `ParameterMapping`。
/// 定义通用参数名 → 厂商特定参数名的映射关系。
///
/// 阶段 1 起由适配器使用；此处仅定义结构，预置常量在适配器层。
#[derive(Debug, Clone, Default)]
pub struct ParameterMapping {
    /// 键名重命名表（值为 None 表示移除该参数）
    pub rename_map: HashMap<String, Option<String>>,
    /// 值映射表：参数名 → {原值键 → 新值}
    ///
    /// 对应 Python v1 `value_map: dict[str, dict[Any, Any]]`。
    /// Python 以原始值（True / 枚举成员等）作 dict 键；Rust 统一为字符串键：
    /// - 字符串值取其本身（枚举式参数的常见情形，如 `"high"`）
    /// - 其余类型取 JSON 序列化文本（`true` → `"true"`、`1` → `"1"`）
    ///
    /// 命中时的行为（对齐 Python v1 `ParameterMapping.apply`）：
    /// - 新值为 JSON 对象 → 展开合并进结果，原参数不再保留
    ///   （如 `reasoning_effort=high` → `{"thinking": {"type": "enabled", ...}}`）
    /// - 新值为非对象 → 替换原值，键名仍按 rename_map 规则处理
    ///   （新值为 Null 时保留键、值为 null，对齐 Python `result[target_key] = None`）
    pub value_map: HashMap<String, HashMap<String, serde_json::Value>>,
}

impl ParameterMapping {
    /// 创建空映射
    pub fn new() -> Self {
        Self::default()
    }

    /// 应用映射规则
    ///
    /// 对齐 Python v1 `ParameterMapping.apply` 的顺序语义：
    /// 1. 先算目标键名（rename_map），显式移除（None）直接跳过，不再查 value_map
    /// 2. 再查 value_map：命中且新值为对象 → 展开合并，原参数不再保留；
    ///    命中且新值为非对象 → 替换原值写入目标键（Null 时保留键、值为 null）
    /// 3. 均未命中 → 原值写入目标键
    pub fn apply(
        &self,
        params: &HashMap<String, serde_json::Value>,
    ) -> HashMap<String, serde_json::Value> {
        let mut result = HashMap::new();
        for (key, value) in params {
            // 键名重命名（None 表示移除，优先于 value_map）
            let target_key = match self.rename_map.get(key) {
                Some(None) => continue,
                Some(Some(new_key)) => new_key.clone(),
                None => key.clone(),
            };

            // 值映射：字符串值取其本身，其余类型取 JSON 序列化文本作查找键
            // （对齐 Python 以 True / 枚举成员等原始值作 dict 键的语义）
            if let Some(values) = self.value_map.get(key) {
                let lookup_key = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                if let Some(mapped) = values.get(&lookup_key) {
                    match mapped {
                        serde_json::Value::Object(map) => {
                            for (k, v) in map {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                        other => {
                            result.insert(target_key, other.clone());
                        }
                    }
                    continue;
                }
            }

            result.insert(target_key, value.clone());
        }
        result
    }
}

/// 文本嵌入请求
///
/// 对应 Python v1 `EmbedOptions` + 嵌入请求参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// 嵌入模型名称
    pub model: String,
    /// 输入文本（单个或多个）
    pub input: EmbedInput,
    /// 输出向量维度（部分模型支持）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// 编码格式（"float" / "base64"）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// 用户标识
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// 厂商特有参数透传
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

/// 嵌入输入（单个字符串或字符串列表）
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbedInput {
    /// 单个文本
    Single(String),
    /// 多个文本
    Multiple(Vec<String>),
}

/// 嵌入结果
///
/// 对应 Python v1 `EmbeddingResult`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    /// 对象类型（固定 "list"）
    #[serde(default = "default_object_list")]
    pub object: String,
    /// 嵌入向量列表
    pub data: Vec<EmbeddingItem>,
    /// 使用的模型
    pub model: String,
    /// 使用统计
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<EmbeddingUsage>,
}

/// 单个嵌入项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingItem {
    /// 对象类型（固定 "embedding"）
    #[serde(default = "default_object_embedding")]
    pub object: String,
    /// 索引
    pub index: u32,
    /// 嵌入向量（浮点列表，或 base64 字符串）
    pub embedding: EmbeddingVector,
}

/// 嵌入向量表示
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    /// 浮点列表
    Float(Vec<f64>),
    /// base64 编码
    Base64(String),
}

/// 嵌入使用统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// 提示词 token 数
    pub prompt_tokens: u64,
    /// 总 token 数
    pub total_tokens: u64,
}

fn default_object_list() -> String {
    "list".into()
}

fn default_object_embedding() -> String {
    "embedding".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_serde() {
        let json = serde_json::to_string(&ReasoningEffort::High).unwrap();
        assert_eq!(json, "\"high\"");
    }

    #[test]
    fn response_format_serde() {
        let json = serde_json::to_string(&ResponseFormat::JsonObject).unwrap();
        assert_eq!(json, "\"json_object\"");
    }

    #[test]
    fn tool_choice_serde() {
        let json = serde_json::to_string(&ToolChoice::Auto).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn stop_seq_untagged() {
        let single = serde_json::to_string(&StopSeq::Single("stop".into())).unwrap();
        assert_eq!(single, "\"stop\"");
        let multi =
            serde_json::to_string(&StopSeq::Multiple(vec!["a".into(), "b".into()])).unwrap();
        assert_eq!(multi, "[\"a\",\"b\"]");
    }

    #[test]
    fn tool_definition_function() {
        let td = ToolDefinition::function(FunctionDefinition {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            parameters: None,
        });
        let json = serde_json::to_string(&td).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"get_weather\""));
    }

    #[test]
    fn tool_definition_web_search() {
        let td = ToolDefinition::web_search();
        let json = serde_json::to_string(&td).unwrap();
        assert!(json.contains("\"type\":\"web_search\""));
    }

    #[test]
    fn tool_call_serde() {
        let tc = ToolCall {
            id: "call_1".into(),
            tool_type: "function".into(),
            function: ToolCallFunction {
                name: "get_weather".into(),
                arguments: "{\"city\":\"Beijing\"}".into(),
            },
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("\"id\":\"call_1\""));
        assert!(json.contains("\"arguments\":\"{\\\"city\\\":\\\"Beijing\\\"}\""));
    }

    #[test]
    fn parameter_mapping_rename() {
        let mut rename = HashMap::new();
        rename.insert("max_tokens".into(), Some("maxOutputTokens".into()));
        rename.insert("web_search".into(), None); // 移除
        let pm = ParameterMapping {
            rename_map: rename,
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("max_tokens".into(), serde_json::json!(1000));
        params.insert("web_search".into(), serde_json::json!(true));
        params.insert("temperature".into(), serde_json::json!(0.7));

        let result = pm.apply(&params);
        assert_eq!(
            result.get("maxOutputTokens").and_then(|v| v.as_i64()),
            Some(1000)
        );
        assert!(!result.contains_key("max_tokens"));
        assert!(!result.contains_key("web_search"));
        assert_eq!(
            result.get("temperature").and_then(|v| v.as_f64()),
            Some(0.7)
        );
    }

    #[test]
    fn parameter_mapping_value_map_expands_object() {
        // 对齐 Python ANTHROPIC_MAPPING：reasoning_effort=high → 展开为 thinking 对象，原键消失
        let mut values = HashMap::new();
        values.insert(
            "high".into(),
            serde_json::json!({"thinking": {"type": "enabled", "budget_tokens": 16384}}),
        );
        let mut value_map = HashMap::new();
        value_map.insert("reasoning_effort".into(), values);
        let pm = ParameterMapping {
            value_map,
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("reasoning_effort".into(), serde_json::json!("high"));
        params.insert("temperature".into(), serde_json::json!(0.7));

        let result = pm.apply(&params);
        assert_eq!(
            result.get("thinking"),
            Some(&serde_json::json!({"type": "enabled", "budget_tokens": 16384}))
        );
        assert!(!result.contains_key("reasoning_effort"));
        // 未命中映射的参数原样透传
        assert_eq!(
            result.get("temperature").and_then(|v| v.as_f64()),
            Some(0.7)
        );
    }

    #[test]
    fn parameter_mapping_value_map_replaces_scalar_value() {
        // 非对象映射值：替换值，键名仍按 rename_map 处理（对齐 Python value = mapped）
        let mut rename = HashMap::new();
        rename.insert("effort".into(), Some("reasoning_effort".into()));
        let mut values = HashMap::new();
        values.insert("high".into(), serde_json::json!("turbo"));
        let mut value_map = HashMap::new();
        value_map.insert("effort".into(), values);
        let pm = ParameterMapping {
            rename_map: rename,
            value_map,
        };

        let mut params = HashMap::new();
        params.insert("effort".into(), serde_json::json!("high"));

        let result = pm.apply(&params);
        assert_eq!(
            result.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("turbo")
        );
        assert!(!result.contains_key("effort"));
    }

    #[test]
    fn parameter_mapping_value_map_null_value_kept() {
        // 映射值为 Null：保留键、值为 null（对齐 Python result[target_key] = None）
        let mut values = HashMap::new();
        values.insert("false".into(), serde_json::Value::Null);
        let mut value_map = HashMap::new();
        value_map.insert("reasoning".into(), values);
        let pm = ParameterMapping {
            value_map,
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("reasoning".into(), serde_json::json!(false));

        let result = pm.apply(&params);
        assert_eq!(result.get("reasoning"), Some(&serde_json::Value::Null));
    }

    #[test]
    fn parameter_mapping_value_map_miss_passes_through() {
        // 值不在映射表中（如 reasoning_effort=none）→ 原样透传
        let mut values = HashMap::new();
        values.insert(
            "high".into(),
            serde_json::json!({"thinking": {"type": "enabled"}}),
        );
        let mut value_map = HashMap::new();
        value_map.insert("reasoning_effort".into(), values);
        let pm = ParameterMapping {
            value_map,
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("reasoning_effort".into(), serde_json::json!("none"));

        let result = pm.apply(&params);
        assert_eq!(
            result.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("none")
        );
        assert!(!result.contains_key("thinking"));
    }

    #[test]
    fn parameter_mapping_value_map_bool_and_number_keys() {
        // 非字符串值按 JSON 序列化形式查表（true → "true"，1 → "1"），
        // 对齐 Python 以 True/枚举成员等原始值作 dict 键的语义
        let mut values = HashMap::new();
        values.insert(
            "true".into(),
            serde_json::json!({"reasoning_effort": "medium"}),
        );
        values.insert("1".into(), serde_json::json!("one"));
        let mut value_map = HashMap::new();
        value_map.insert("reasoning".into(), values);
        let pm = ParameterMapping {
            value_map,
            ..Default::default()
        };

        let mut params = HashMap::new();
        params.insert("reasoning".into(), serde_json::json!(true));
        let result = pm.apply(&params);
        assert_eq!(
            result.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("medium")
        );
        assert!(!result.contains_key("reasoning"));

        let mut params2 = HashMap::new();
        params2.insert("reasoning".into(), serde_json::json!(1));
        let result2 = pm.apply(&params2);
        assert_eq!(
            result2.get("reasoning").and_then(|v| v.as_str()),
            Some("one")
        );
    }

    #[test]
    fn parameter_mapping_rename_remove_takes_precedence_over_value_map() {
        // rename_map 移除（None）优先于 value_map
        // （对齐 Python：先算 target_key，为 None 直接 continue，不再查 value_map）
        let mut rename = HashMap::new();
        rename.insert("reasoning".into(), None);
        let mut values = HashMap::new();
        values.insert(
            "true".into(),
            serde_json::json!({"reasoning_effort": "medium"}),
        );
        let mut value_map = HashMap::new();
        value_map.insert("reasoning".into(), values);
        let pm = ParameterMapping {
            rename_map: rename,
            value_map,
        };

        let mut params = HashMap::new();
        params.insert("reasoning".into(), serde_json::json!(true));

        let result = pm.apply(&params);
        assert!(result.is_empty());
    }

    #[test]
    fn embed_input_serde() {
        let single = EmbedInput::Single("hello".into());
        let json = serde_json::to_string(&single).unwrap();
        assert_eq!(json, "\"hello\"");

        let multi = EmbedInput::Multiple(vec!["a".into(), "b".into()]);
        let json = serde_json::to_string(&multi).unwrap();
        assert_eq!(json, "[\"a\",\"b\"]");
    }

    #[test]
    fn embedding_result_serde() {
        let r = EmbeddingResult {
            object: "list".into(),
            data: vec![EmbeddingItem {
                object: "embedding".into(),
                index: 0,
                embedding: EmbeddingVector::Float(vec![0.1, 0.2, 0.3]),
            }],
            model: "text-embedding-3-small".into(),
            usage: Some(EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"model\":\"text-embedding-3-small\""));
    }

    #[test]
    fn embedding_vector_base64() {
        let v = EmbeddingVector::Base64("aGVsbG8=".into());
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"aGVsbG8=\"");
    }

    #[test]
    fn embed_request_skip_empty_extra() {
        let req = EmbedRequest {
            model: "m".into(),
            input: EmbedInput::Single("hi".into()),
            dimensions: None,
            encoding_format: None,
            user: None,
            extra: HashMap::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("extra"));
        assert!(!json.contains("dimensions"));
    }
}
