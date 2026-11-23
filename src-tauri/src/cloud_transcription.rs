//! Transcription through a remote OpenAI-compatible API instead of a local
//! model.
//!
//! Two request shapes are supported (see [`CloudApiMode`]):
//!
//! * **Chat audio** — `POST {base}/chat/completions` with the recording as an
//!   `input_audio` content part. An audio-capable chat model (e.g. OpenAI's
//!   `gpt-4o-mini-audio-preview`, Gemini through its OpenAI-compatible
//!   endpoint, or audio models on OpenRouter) hears the audio and returns the
//!   text. When post-processing is requested, the selected post-processing
//!   prompt is folded into the same request, so speech-to-text and the cleanup
//!   happen in one round trip.
//! * **Transcriptions** — `POST {base}/audio/transcriptions` (multipart), the
//!   Whisper-style endpoint (`whisper-1`, `gpt-4o-transcribe`, Groq, a local
//!   speaches/faster-whisper server, ...). Post-processing, if requested, runs
//!   afterwards through the regular post-processing provider.

use crate::settings::{AppSettings, CloudApiMode, CloudTranscriptionSettings};
use base64::Engine as _;
use log::{debug, warn};
use serde_json::{json, Value};
use std::io::Cursor;
use std::sync::OnceLock;
use std::time::Duration;

/// Sample rate of every recording Handy hands to a transcriber.
const SAMPLE_RATE: u32 = crate::audio_toolkit::constants::WHISPER_SAMPLE_RATE;

/// Deadline for one cloud request. Uploading a long recording on a slow link
/// plus the model's own time can take a while, but a dictation should never
/// hang forever on an endpoint that accepted the request and went quiet.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// Placeholder put where a post-processing prompt expects `${output}`: in chat
/// mode there is no transcript yet, the model hears the audio instead.
const AUDIO_TRANSCRIPT_PLACEHOLDER: &str = "(the transcript of the attached audio recording)";

/// The result of one cloud transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudTranscription {
    pub text: String,
    /// The post-processing prompt the model already applied in the same
    /// request (chat mode only). `None` means `text` is a plain transcript.
    pub applied_post_process_prompt: Option<String>,
}

/// Why a cloud transcription failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudError {
    pub message: String,
    /// The API couldn't be reached (no network, DNS failure, timeout, or a
    /// gateway error), as opposed to rejecting the request. Only these fall
    /// back to the local model.
    pub unreachable: bool,
}

impl CloudError {
    fn unreachable(message: String) -> Self {
        Self {
            message,
            unreachable: true,
        }
    }
}

impl From<String> for CloudError {
    fn from(message: String) -> Self {
        Self {
            message,
            unreachable: false,
        }
    }
}

impl std::fmt::Display for CloudError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CloudError {}

/// Whether cloud transcription replaces the local model.
pub fn is_enabled(settings: &AppSettings) -> bool {
    settings.cloud_transcription.enabled
}

/// Whether a failed cloud request may be retried with the local model: the
/// user allows it and a local model is selected.
pub fn can_fall_back(settings: &AppSettings) -> bool {
    settings.cloud_transcription.fallback_to_local && !settings.selected_model.is_empty()
}

/// A quick guess whether the API is reachable right now: any HTTP answer from
/// its base URL within a few seconds counts. Used at recording start to load
/// the local model early when the fallback will likely be needed.
pub async fn is_reachable(settings: &AppSettings) -> bool {
    let cloud = &settings.cloud_transcription;
    let Ok(client) = client() else {
        return false;
    };
    let url = endpoint(cloud, "models");
    match with_auth(client.get(url), cloud)
        .timeout(Duration::from_secs(4))
        .send()
        .await
    {
        Ok(_) => true,
        Err(e) => {
            debug!(
                "Cloud API looks unreachable: {}",
                describe_reqwest_error(e).message
            );
            false
        }
    }
}

/// Transcribes 16 kHz mono `samples` with the configured cloud API.
/// `post_process` asks for the selected post-processing prompt to be applied
/// too; in chat mode that happens in the same request.
pub async fn transcribe(
    settings: &AppSettings,
    samples: &[f32],
    post_process: bool,
) -> Result<CloudTranscription, CloudError> {
    let cloud = &settings.cloud_transcription;
    if cloud.base_url.trim().is_empty() {
        return Err("Cloud transcription has no API base URL".to_string().into());
    }
    if cloud.model.trim().is_empty() {
        return Err("Cloud transcription has no model".to_string().into());
    }
    if samples.is_empty() {
        return Ok(CloudTranscription {
            text: String::new(),
            applied_post_process_prompt: None,
        });
    }

    let wav = encode_wav(samples)?;
    debug!(
        "Cloud transcription: {:?} mode, model '{}', {:.1}s of audio ({} KiB WAV)",
        cloud.api_mode,
        cloud.model,
        samples.len() as f64 / SAMPLE_RATE as f64,
        wav.len() / 1024
    );

    match cloud.api_mode {
        CloudApiMode::ChatAudio => {
            let post_prompt = if post_process || cloud.always_post_process {
                selected_post_process_prompt(settings)
            } else {
                None
            };
            if (post_process || cloud.always_post_process) && post_prompt.is_none() {
                warn!("Cloud post-processing requested but no post-processing prompt is selected");
            }
            let system_prompt = chat_system_prompt(settings, post_prompt.as_deref());
            let text = chat_audio_request(cloud, &system_prompt, &wav).await?;
            Ok(CloudTranscription {
                text,
                applied_post_process_prompt: post_prompt,
            })
        }
        CloudApiMode::Transcriptions => {
            let text = transcriptions_request(settings, &wav).await?;
            Ok(CloudTranscription {
                text,
                applied_post_process_prompt: None,
            })
        }
    }
}

/// The selected post-processing prompt's text, if one is selected and not blank.
fn selected_post_process_prompt(settings: &AppSettings) -> Option<String> {
    let id = settings.post_process_selected_prompt_id.as_ref()?;
    settings
        .post_process_prompts
        .iter()
        .find(|p| &p.id == id)
        .map(|p| p.prompt.clone())
        .filter(|p| !p.trim().is_empty())
}

/// Language code to hint to the model, or `None` for automatic detection.
fn language_hint(settings: &AppSettings) -> Option<String> {
    let lang = settings.selected_language.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(lang.to_string())
    }
}

/// System prompt for chat mode: the transcription instructions, the user's
/// vocabulary and language hints, then the post-processing instructions.
pub(crate) fn chat_system_prompt(settings: &AppSettings, post_prompt: Option<&str>) -> String {
    let cloud = &settings.cloud_transcription;
    let mut prompt = if cloud.prompt.trim().is_empty() {
        crate::settings::DEFAULT_CLOUD_TRANSCRIPTION_PROMPT.to_string()
    } else {
        cloud.prompt.trim().to_string()
    };

    if let Some(lang) = language_hint(settings) {
        prompt.push_str(&format!(
            "\n\nThe recording is spoken in the language with code \"{lang}\"."
        ));
    }
    if settings.translate_to_english {
        prompt.push_str(
            "\n\nTranslate the transcript into English and output only the English text.",
        );
    }
    if !settings.custom_words.is_empty() {
        prompt.push_str(&format!(
            "\n\nThese words or names may occur in the recording; spell them exactly like this when they do: {}.",
            settings.custom_words.join(", ")
        ));
    }

    if let Some(post_prompt) = post_prompt {
        let post_prompt = post_prompt.replace("${output}", AUDIO_TRANSCRIPT_PLACEHOLDER);
        prompt.push_str(
            "\n\nAfter transcribing, process the transcript according to the instructions \
             below. Output ONLY the processed result, not the raw transcript, and nothing else.\n\n\
             --- Post-processing instructions ---\n",
        );
        prompt.push_str(post_prompt.trim());
    }

    prompt
}

/// Encodes 16 kHz mono float samples as a 16-bit PCM WAV file in memory.
pub(crate) fn encode_wav(samples: &[f32]) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("Failed to encode audio: {e}"))?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(v)
                .map_err(|e| format!("Failed to encode audio: {e}"))?;
        }
        writer
            .finalize()
            .map_err(|e| format!("Failed to encode audio: {e}"))?;
    }
    Ok(cursor.into_inner())
}

fn client() -> Result<&'static reqwest::Client, String> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if let Some(c) = CLIENT.get() {
        return Ok(c);
    }
    let c = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(15))
        .connect_timeout(Duration::from_secs(10))
        .timeout(REQUEST_TIMEOUT)
        .user_agent("Handy/1.0 (+https://github.com/cjpais/Handy)")
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;
    Ok(CLIENT.get_or_init(|| c))
}

fn endpoint(cloud: &CloudTranscriptionSettings, path: &str) -> String {
    format!("{}/{}", cloud.base_url.trim().trim_end_matches('/'), path)
}

fn with_auth(
    req: reqwest::RequestBuilder,
    cloud: &CloudTranscriptionSettings,
) -> reqwest::RequestBuilder {
    let key = cloud.api_key.trim();
    if key.is_empty() {
        req
    } else {
        req.bearer_auth(key)
    }
}

/// Describes a transport failure without echoing the request URL's query.
/// Every transport failure means the API wasn't reached (or stopped
/// answering), so all of them count as unreachable.
fn describe_reqwest_error(e: reqwest::Error) -> CloudError {
    CloudError::unreachable(if e.is_timeout() {
        "The cloud API did not answer in time".to_string()
    } else if e.is_connect() {
        "Could not connect to the cloud API (check the base URL and your network)".to_string()
    } else {
        format!("Cloud API request failed: {}", e.without_url())
    })
}

/// Turns a non-success HTTP answer into a short, readable error. Gateway
/// errors (502/503/504) mean the service is down, so they count as unreachable.
fn http_error(status: reqwest::StatusCode, body: &str) -> CloudError {
    let message = http_error_message(status, body);
    let gateway = matches!(status.as_u16(), 502..=504);
    CloudError {
        message,
        unreachable: gateway,
    }
}

fn http_error_message(status: reqwest::StatusCode, body: &str) -> String {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message")
                .or_else(|| v.get("error"))
                .or_else(|| v.get("message"))
                .or_else(|| v.get("detail"))
                .map(|m| match m {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
        })
        .unwrap_or_else(|| body.trim().to_string());
    let detail: String = detail.chars().take(300).collect();
    if detail.is_empty() {
        format!("Cloud API returned HTTP {status}")
    } else {
        format!("Cloud API returned HTTP {status}: {detail}")
    }
}

/// `/chat/completions` with the recording attached as `input_audio`.
async fn chat_audio_request(
    cloud: &CloudTranscriptionSettings,
    system_prompt: &str,
    wav: &[u8],
) -> Result<String, CloudError> {
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(wav);
    let body = json!({
        "model": cloud.model.trim(),
        "stream": false,
        "messages": [
            { "role": "system", "content": system_prompt },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Here is the audio recording." },
                    {
                        "type": "input_audio",
                        "input_audio": { "data": audio_b64, "format": "wav" }
                    }
                ]
            }
        ]
    });

    let response = with_auth(client()?.post(endpoint(cloud, "chat/completions")), cloud)
        .json(&body)
        .send()
        .await
        .map_err(describe_reqwest_error)?;
    let status = response.status();
    let text = response.text().await.map_err(describe_reqwest_error)?;
    if !status.is_success() {
        return Err(http_error(status, &text));
    }
    let json: Value = serde_json::from_str(&text)
        .map_err(|_| "Cloud API returned a response that is not JSON".to_string())?;
    parse_chat_content(&json)
        .map(|c| clean_output(&c))
        .ok_or_else(|| {
            "Cloud API response has no message content"
                .to_string()
                .into()
        })
}

/// The assistant message text of a chat completion. Content may be a string
/// or, on some providers, a list of typed parts.
pub(crate) fn parse_chat_content(json: &Value) -> Option<String> {
    let content = json.pointer("/choices/0/message/content")?;
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect();
            Some(text)
        }
        Value::Null => {
            // Some audio models answer in `message.audio.transcript`.
            json.pointer("/choices/0/message/audio/transcript")
                .and_then(Value::as_str)
                .map(str::to_string)
        }
        _ => None,
    }
}

/// `/audio/transcriptions`, Whisper-style multipart upload.
async fn transcriptions_request(settings: &AppSettings, wav: &[u8]) -> Result<String, CloudError> {
    let cloud = &settings.cloud_transcription;
    let file = reqwest::multipart::Part::bytes(wav.to_vec())
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("Failed to build the upload: {e}"))?;
    let mut form = reqwest::multipart::Form::new()
        .part("file", file)
        .text("model", cloud.model.trim().to_string())
        .text("response_format", "json");

    // The Whisper `prompt` biases spelling and style; the default chat prompt
    // is an instruction, which Whisper models would only echo, so it's left out.
    let mut prompt = String::new();
    if !cloud.prompt.trim().is_empty()
        && cloud.prompt.trim() != crate::settings::DEFAULT_CLOUD_TRANSCRIPTION_PROMPT
    {
        prompt.push_str(cloud.prompt.trim());
    }
    if !settings.custom_words.is_empty() {
        if !prompt.is_empty() {
            prompt.push(' ');
        }
        prompt.push_str(&settings.custom_words.join(", "));
    }
    if !prompt.is_empty() {
        form = form.text("prompt", prompt);
    }
    if let Some(lang) = language_hint(settings) {
        // ISO-639-1: "zh-Hans" → "zh".
        let code = lang.split(['-', '_']).next().unwrap_or("").to_lowercase();
        if code.len() == 2 {
            form = form.text("language", code);
        }
    }

    let response = with_auth(
        client()?.post(endpoint(cloud, "audio/transcriptions")),
        cloud,
    )
    .multipart(form)
    .send()
    .await
    .map_err(describe_reqwest_error)?;
    let status = response.status();
    let text = response.text().await.map_err(describe_reqwest_error)?;
    if !status.is_success() {
        return Err(http_error(status, &text));
    }
    // `response_format=json` gives {"text": ...}; tolerate servers that
    // answer with plain text anyway.
    match serde_json::from_str::<Value>(&text) {
        Ok(json) => json
            .get("text")
            .and_then(Value::as_str)
            .map(clean_output)
            .ok_or_else(|| {
                "Cloud API response has no \"text\" field"
                    .to_string()
                    .into()
            }),
        Err(_) => Ok(clean_output(&text)),
    }
}

/// Trims the model output and drops a leading `<think>` block and invisible
/// characters some models emit.
pub(crate) fn clean_output(s: &str) -> String {
    let mut s = s.trim_start();
    if let Some(rest) = s.strip_prefix("<think>") {
        if let Some(end) = rest.find("</think>") {
            s = rest[end + "</think>".len()..].trim_start();
        }
    }
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::get_default_settings;

    #[test]
    fn wav_has_header_and_16_bit_samples() {
        let wav = encode_wav(&[0.0, 0.5, -0.5, 1.0]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + 4 * 2);
    }

    #[test]
    fn chat_content_string_and_parts() {
        let a = json!({"choices":[{"message":{"content":"hello"}}]});
        assert_eq!(parse_chat_content(&a).as_deref(), Some("hello"));
        let b = json!({"choices":[{"message":{"content":[{"type":"text","text":"he"},{"type":"text","text":"llo"}]}}]});
        assert_eq!(parse_chat_content(&b).as_deref(), Some("hello"));
        let c = json!({"choices":[{"message":{"content":null,"audio":{"transcript":"hi"}}}]});
        assert_eq!(parse_chat_content(&c).as_deref(), Some("hi"));
        let d = json!({"choices":[]});
        assert_eq!(parse_chat_content(&d), None);
    }

    #[test]
    fn output_is_cleaned() {
        assert_eq!(
            clean_output("  <think>hmm</think>\n Hello\u{200B} "),
            "Hello"
        );
        assert_eq!(clean_output("plain"), "plain");
    }

    #[test]
    fn system_prompt_folds_in_post_processing() {
        let mut settings = get_default_settings();
        settings.custom_words = vec!["Handy".into(), "Tauri".into()];
        settings.selected_language = "de".into();
        let prompt = chat_system_prompt(&settings, Some("<t>\n${output}\n</t>\nClean it."));
        assert!(prompt.starts_with("You are a speech-to-text engine."));
        assert!(prompt.contains("\"de\""));
        assert!(prompt.contains("Handy, Tauri"));
        assert!(prompt.contains(AUDIO_TRANSCRIPT_PLACEHOLDER));
        assert!(prompt.ends_with("Clean it."));
        assert!(!prompt.contains("${output}"));

        let plain = chat_system_prompt(&get_default_settings(), None);
        assert!(!plain.contains("Post-processing"));
    }

    #[test]
    fn http_errors_are_readable() {
        let e = http_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Incorrect API key provided"}}"#,
        );
        assert_eq!(
            e.message,
            "Cloud API returned HTTP 401 Unauthorized: Incorrect API key provided"
        );
        assert!(!e.unreachable);
        assert!(http_error(reqwest::StatusCode::SERVICE_UNAVAILABLE, "").unreachable);
    }

    /// One-shot HTTP server answering every request with `body`.
    fn serve_once(body: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = vec![0u8; 1 << 20];
                // Read until the headers and declared body length arrived.
                let mut read = 0;
                loop {
                    let n = stream.read(&mut buf[read..]).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    read += n;
                    let head = String::from_utf8_lossy(&buf[..read]);
                    if let Some(end) = head.find("\r\n\r\n") {
                        let len = head[..end]
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if read >= end + 4 + len {
                            break;
                        }
                    }
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://{addr}/v1")
    }

    #[test]
    fn chat_mode_round_trip_applies_post_processing() {
        let url = serve_once(
            r#"{"choices":[{"message":{"role":"assistant","content":"Hello, world."}}]}"#,
        );
        let mut settings = get_default_settings();
        settings.cloud_transcription.enabled = true;
        settings.cloud_transcription.base_url = url;
        settings.post_process_selected_prompt_id =
            settings.post_process_prompts.first().map(|p| p.id.clone());
        let result =
            tauri::async_runtime::block_on(transcribe(&settings, &[0.1; 1600], true)).unwrap();
        assert_eq!(result.text, "Hello, world.");
        assert!(result.applied_post_process_prompt.is_some());
    }

    #[test]
    fn an_unreachable_api_is_reported_as_such() {
        // Nothing listens on port 1.
        let mut settings = get_default_settings();
        settings.cloud_transcription.enabled = true;
        settings.cloud_transcription.base_url = "http://127.0.0.1:1/v1".into();
        let err =
            tauri::async_runtime::block_on(transcribe(&settings, &[0.1; 1600], false)).unwrap_err();
        assert!(err.unreachable, "{err}");
        assert!(!tauri::async_runtime::block_on(is_reachable(&settings)));

        // Falling back needs a local model.
        settings.selected_model.clear();
        assert!(!can_fall_back(&settings));
        settings.selected_model = "parakeet-tdt-0.6b-v3".into();
        assert!(can_fall_back(&settings));
        settings.cloud_transcription.fallback_to_local = false;
        assert!(!can_fall_back(&settings));
    }

    #[test]
    fn transcriptions_mode_round_trip() {
        let url = serve_once(r#"{"text":" hello there "}"#);
        let mut settings = get_default_settings();
        settings.cloud_transcription.enabled = true;
        settings.cloud_transcription.api_mode = CloudApiMode::Transcriptions;
        settings.cloud_transcription.model = "whisper-1".into();
        settings.cloud_transcription.base_url = url;
        let result =
            tauri::async_runtime::block_on(transcribe(&settings, &[0.1; 1600], true)).unwrap();
        assert_eq!(result.text, "hello there");
        assert_eq!(result.applied_post_process_prompt, None);
    }
}
