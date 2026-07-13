'use strict';

// AIBridge Node.js 绑定入口（含 JS 侧包装）
//
// napi build 会自动生成 index.js（纯 native re-export），因此本文件作为
// package.json 的 main 入口，负责：
// 1. 加载 native 模块
// 2. 包装 Client 的所有方法与构造函数，统一为抛出的 Error 解析出 `.code` 属性
//    （Rust 侧 map_error 将 AibridgeError 编码为 `[code] message`）
// 3. 为 ChatStreamIterator.prototype 安装 [Symbol.asyncIterator]，支持 `for await...of`

const native = require('./index.js');
const NativeClient = native.Client;
const { ChatStreamIterator } = native;

/**
 * 错误码前缀正则：匹配 `[code] message` 格式
 */
const ERROR_CODE_RE = /^\[([a-z_]+)\]\s*(.*)$/s;

/**
 * 包装 Error，解析出 `.code` 属性
 *
 * Rust 侧 map_error 将 AibridgeError 编码为 `[code] message`，
 * 此处解析出 code 并挂到 Error.code 属性上；无前缀的视为 unknown_error。
 * @param {Error} err - 原始错误
 * @returns {Error} 带 .code 属性的错误
 */
function withCode(err) {
  if (err && typeof err.message === 'string') {
    const m = err.message.match(ERROR_CODE_RE);
    if (m) {
      err.code = m[1];
      err.message = m[2];
    } else if (!err.code) {
      err.code = 'unknown_error';
    }
  }
  return err;
}

/**
 * 包装一个返回 Promise 的方法，reject 时解析 .code
 */
function wrapAsync(fn) {
  return function (...args) {
    const p = fn.apply(this, args);
    if (p && typeof p.then === 'function') {
      return p.catch((err) => Promise.reject(withCode(err)));
    }
    return p;
  };
}

/**
 * AIBridge 统一客户端（JS 包装层）
 *
 * 代理原生 Client，构造与方法调用的错误统一经 withCode 解析出 `.code`。
 * 其余行为与原生 Client 完全一致。
 */
class Client {
  constructor(provider, options) {
    try {
      this._native = new NativeClient(provider, options);
    } catch (err) {
      throw withCode(err);
    }
  }

  start() {
    return wrapAsync(() => this._native.start()).call(this);
  }

  close() {
    return wrapAsync(() => this._native.close()).call(this);
  }

  chat(request) {
    return wrapAsync(() => this._native.chat(request)).call(this);
  }

  speech(request) {
    return wrapAsync(() => this._native.speech(request)).call(this);
  }

  chatStream(request) {
    // chatStream 返回 Promise<ChatStreamIterator>，reject 时需解析 code
    return this._native.chatStream(request).catch((err) =>
      Promise.reject(withCode(err))
    );
  }
}

/**
 * 为 ChatStreamIterator.prototype 安装 [Symbol.asyncIterator]
 *
 * napi 2 的 Generator trait 是同步的，无法桥接异步 stream，因此通过 next() 方法
 * （返回 Promise<chunk|null>）手动实现 asyncIterator 协议。
 */
if (ChatStreamIterator && !ChatStreamIterator.prototype[Symbol.asyncIterator]) {
  ChatStreamIterator.prototype[Symbol.asyncIterator] = function asyncIterator() {
    const self = this;
    return {
      async next() {
        try {
          const chunk = await self.next();
          // chunk === null 表示流结束
          if (chunk === null || chunk === undefined) {
            return { value: undefined, done: true };
          }
          return { value: chunk, done: false };
        } catch (err) {
          // 流内部错误，解析 code 后抛出
          throw withCode(err);
        }
      },
    };
  };
}

module.exports = { Client, ChatStreamIterator };
module.exports.default = module.exports;
module.exports.Client = Client;
module.exports.ChatStreamIterator = ChatStreamIterator;
