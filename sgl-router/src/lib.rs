// SGLang路由器库主模块
// 这个库实现了SGLang的负载均衡路由器，用于在多个推理服务器之间分发请求

use pyo3::prelude::*;  // PyO3 Python绑定库
pub mod config;  // 配置模块
pub mod logging;  // 日志模块
use std::collections::HashMap;  // 哈希映射集合
pub mod core;  // 核心功能模块
pub mod metrics;  // 指标收集模块
pub mod middleware;  // 中间件模块
pub mod openai_api_types;  // OpenAI API类型定义
pub mod policies;  // 负载均衡策略模块
pub mod routers;  // 路由器实现模块
pub mod server;  // 服务器模块
pub mod service_discovery;  // 服务发现模块
pub mod tree;  // 树结构模块
use crate::metrics::PrometheusConfig;  // Prometheus配置

/// 负载均衡策略类型枚举
/// 定义了路由器支持的不同负载均衡策略
#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,      // 随机选择策略
    RoundRobin,  // 轮询策略
    CacheAware,  // 缓存感知策略
    PowerOfTwo,  // 二次幂选择策略
}

/// 路由器结构体
/// 包含了路由器的所有配置参数和状态信息
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    host: String,  // 路由器主机地址
    port: u16,  // 路由器端口号
    worker_urls: Vec<String>,  // 工作节点URL列表
    policy: PolicyType,  // 负载均衡策略类型
    worker_startup_timeout_secs: u64,  // 工作节点启动超时时间（秒）
    worker_startup_check_interval: u64,  // 工作节点启动检查间隔（秒）
    cache_threshold: f32,  // 缓存阈值
    balance_abs_threshold: usize,  // 绝对负载均衡阈值
    balance_rel_threshold: f32,  // 相对负载均衡阈值
    eviction_interval_secs: u64,  // 缓存驱逐间隔（秒）
    max_tree_size: usize,  // 最大树大小
    max_payload_size: usize,  // 最大负载大小
    dp_aware: bool,  // 是否启用数据并行感知
    api_key: Option<String>,  // API密钥（可选）
    log_dir: Option<String>,  // 日志目录（可选）
    log_level: Option<String>,  // 日志级别（可选）
    service_discovery: bool,  // 是否启用服务发现
    selector: HashMap<String, String>,  // 选择器配置
    service_discovery_port: u16,  // 服务发现端口
    service_discovery_namespace: Option<String>,  // 服务发现命名空间（可选）
    prefill_selector: HashMap<String, String>,  // 预填充选择器
    decode_selector: HashMap<String, String>,  // 解码选择器
    bootstrap_port_annotation: String,  // 引导端口注解
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    request_timeout_secs: u64,
    request_id_headers: Option<Vec<String>>,
    pd_disaggregation: bool,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: usize,
    cors_allowed_origins: Vec<String>,
    // Retry configuration
    retry_max_retries: u32,
    retry_initial_backoff_ms: u64,
    retry_max_backoff_ms: u64,
    retry_backoff_multiplier: f32,
    retry_jitter_factor: f32,
    disable_retries: bool,
    // Circuit breaker configuration
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    disable_circuit_breaker: bool,
}

impl Router {
    /// Convert PyO3 Router to RouterConfig
    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, MetricsConfig, PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        // Convert policy helper function
        let convert_policy = |policy: &PolicyType| -> ConfigPolicyConfig {
            match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5, // Default value
                },
            }
        };

        // Determine routing mode
        let mode = if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self.prefill_policy.as_ref().map(convert_policy),
                decode_policy: self.decode_policy.as_ref().map(convert_policy),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        // Convert main policy
        let policy = convert_policy(&self.policy);

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = match (self.prometheus_port, self.prometheus_host.as_ref()) {
            (Some(port), Some(host)) => Some(MetricsConfig {
                port,
                host: host.clone(),
            }),
            _ => None,
        };

        Ok(config::RouterConfig {
            mode,
            policy,
            host: self.host.clone(),
            port: self.port,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            dp_aware: self.dp_aware,
            api_key: self.api_key.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: self.log_level.clone(),
            request_id_headers: self.request_id_headers.clone(),
            max_concurrent_requests: self.max_concurrent_requests,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: false,
            disable_circuit_breaker: false,
        })
    }
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        worker_startup_timeout_secs = 300,
        worker_startup_check_interval = 10,
        cache_threshold = 0.50,
        balance_abs_threshold = 32,
        balance_rel_threshold = 1.0001,
        eviction_interval_secs = 60,
        max_tree_size = 2usize.pow(24),
        max_payload_size = 256 * 1024 * 1024,  // 256MB default for large batches
        dp_aware = false,
        api_key = None,
        log_dir = None,
        log_level = None,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prefill_selector = HashMap::new(),
        decode_selector = HashMap::new(),
        bootstrap_port_annotation = String::from("sglang.ai/bootstrap-port"),
        prometheus_port = None,
        prometheus_host = None,
        request_timeout_secs = 600,  // Add configurable request timeout
        request_id_headers = None,  // Custom request ID headers
        pd_disaggregation = false,  // New flag for PD mode
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = 64,
        cors_allowed_origins = vec![],
        // Retry defaults
        retry_max_retries = 3,
        retry_initial_backoff_ms = 100,
        retry_max_backoff_ms = 10_000,
        retry_backoff_multiplier = 2.0,
        retry_jitter_factor = 0.1,
        disable_retries = false,
        // Circuit breaker defaults
        cb_failure_threshold = 5,
        cb_success_threshold = 2,
        cb_timeout_duration_secs = 30,
        cb_window_duration_secs = 60,
        disable_circuit_breaker = false,
    ))]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_payload_size: usize,
        dp_aware: bool,
        api_key: Option<String>,
        log_dir: Option<String>,
        log_level: Option<String>,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prefill_selector: HashMap<String, String>,
        decode_selector: HashMap<String, String>,
        bootstrap_port_annotation: String,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
        request_timeout_secs: u64,
        request_id_headers: Option<Vec<String>>,
        pd_disaggregation: bool,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: usize,
        cors_allowed_origins: Vec<String>,
        retry_max_retries: u32,
        retry_initial_backoff_ms: u64,
        retry_max_backoff_ms: u64,
        retry_backoff_multiplier: f32,
        retry_jitter_factor: f32,
        disable_retries: bool,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        disable_circuit_breaker: bool,
    ) -> PyResult<Self> {
        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            max_payload_size,
            dp_aware,
            api_key,
            log_dir,
            log_level,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation,
            prometheus_port,
            prometheus_host,
            request_timeout_secs,
            request_id_headers,
            pd_disaggregation,
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
            max_concurrent_requests,
            cors_allowed_origins,
            retry_max_retries,
            retry_initial_backoff_ms,
            retry_max_backoff_ms,
            retry_backoff_multiplier,
            retry_jitter_factor,
            disable_retries,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            disable_circuit_breaker,
        })
    }

    fn start(&self) -> PyResult<()> {
        // Convert to RouterConfig and validate
        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {}", e))
        })?;

        // Validate the configuration
        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Configuration validation failed: {}",
                e
            ))
        })?;

        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.pd_disaggregation,
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        // Create Prometheus config if enabled
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
        });

        // Use tokio runtime instead of actix-web System for better compatibility
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Block on the async startup function
        runtime.block_on(async move {
            server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                router_config,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                log_level: self.log_level.clone(),
                service_discovery_config,
                prometheus_config,
                request_timeout_secs: self.request_timeout_secs,
                request_id_headers: self.request_id_headers.clone(),
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

#[pymodule]
fn sglang_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    Ok(())
}
