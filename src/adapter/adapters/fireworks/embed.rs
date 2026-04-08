//! Fireworks Embeddings API implementation.
//!
//! Notes:
//! - Public user-facing shorthand like `fireworks::qwen3-embedding-8b` is
//!   normalized by the adapter to `accounts/fireworks/models/qwen3-embedding-8b`.

use crate::adapter::adapters::support::get_api_key;
use crate::adapter::{Adapter, ServiceType, WebRequestData};
use crate::chat::Usage;
use crate::embed::{EmbedOptionsSet, EmbedRequest, EmbedResponse, Embedding};
use crate::webc::WebResponse;
use crate::{Error, Headers, ModelIden, Result, ServiceTarget};
use serde::{Deserialize, Serialize};

use super::FireworksAdapter;

// region:    --- Fireworks Embed Request

#[derive(Debug, Serialize)]
struct FireworksEmbedRequest {
	input: FireworksEmbedInput,
	model: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	encoding_format: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	dimensions: Option<usize>,
	#[serde(skip_serializing_if = "Option::is_none")]
	user: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum FireworksEmbedInput {
	Single(String),
	Batch(Vec<String>),
}

// endregion: --- Fireworks Embed Request

// region:    --- Fireworks Embed Response

#[derive(Debug, Deserialize)]
struct FireworksEmbedResponse {
	data: Vec<FireworksEmbedData>,
	model: String,
	usage: FireworksEmbedUsage,
}

#[derive(Debug, Deserialize)]
struct FireworksEmbedData {
	embedding: Vec<f32>,
	index: usize,
}

#[derive(Debug, Deserialize)]
struct FireworksEmbedUsage {
	prompt_tokens: u32,
	total_tokens: u32,
}

// endregion: --- Fireworks Embed Response

// region:    --- Public Functions

pub fn to_embed_request_data(
	mut service_target: ServiceTarget,
	embed_req: EmbedRequest,
	options_set: EmbedOptionsSet<'_, '_>,
) -> Result<WebRequestData> {
	// Normalize namespaced shorthand to the Fireworks canonical model id used by
	// the embeddings endpoint, e.g. `fireworks::qwen3-embedding-8b` ->
	// `accounts/fireworks/models/qwen3-embedding-8b`.
	if !service_target.model.model_name.contains('/') {
		service_target.model = service_target.model.from_name(format!(
			"accounts/fireworks/models/{}",
			service_target.model.model_name.namespace_and_name().1
		))
	}
	let ServiceTarget { model, auth, .. } = service_target;
	let api_key = get_api_key(auth, &model)?;

	let mut headers = Headers::from(vec![
		("Accept".to_string(), "application/json".to_string()),
		("Authorization".to_string(), format!("Bearer {api_key}")),
		("Content-Type".to_string(), "application/json".to_string()),
	]);

	if let Some(custom_headers) = options_set.headers() {
		headers.merge_with(custom_headers);
	}

	let input = match embed_req.input {
		crate::embed::EmbedInput::Single(text) => FireworksEmbedInput::Single(text),
		crate::embed::EmbedInput::Batch(texts) => FireworksEmbedInput::Batch(texts),
	};

	let (_, model_name) = model.model_name.namespace_and_name();

	let req = FireworksEmbedRequest {
		input,
		model: model_name.to_string(),
		encoding_format: options_set.encoding_format().map(|s| s.to_string()),
		dimensions: options_set.dimensions(),
		user: options_set.user().map(|s| s.to_string()),
	};

	let payload = serde_json::to_value(req).map_err(|serde_error| Error::StreamParse {
		model_iden: model.clone(),
		serde_error,
	})?;
	let url = <FireworksAdapter as Adapter>::get_service_url(&model, ServiceType::Embed, service_target.endpoint)?;

	Ok(WebRequestData { url, headers, payload })
}

pub fn to_embed_response(
	model_iden: ModelIden,
	web_response: WebResponse,
	options_set: EmbedOptionsSet<'_, '_>,
) -> Result<EmbedResponse> {
	let WebResponse { body, .. } = web_response;

	let fireworks_res: FireworksEmbedResponse =
		serde_json::from_value(body.clone()).map_err(|serde_error| Error::StreamParse {
			model_iden: model_iden.clone(),
			serde_error,
		})?;

	let embeddings: Vec<Embedding> = fireworks_res
		.data
		.into_iter()
		.map(|data| Embedding::new(data.embedding, data.index))
		.collect();

	let usage = Usage {
		prompt_tokens: Some(fireworks_res.usage.prompt_tokens as i32),
		completion_tokens: None,
		total_tokens: Some(fireworks_res.usage.total_tokens as i32),
		prompt_tokens_details: None,
		completion_tokens_details: None,
	};

	let provider_model_iden = ModelIden {
		adapter_kind: model_iden.adapter_kind,
		model_name: fireworks_res.model.into(),
	};

	let mut response = EmbedResponse::new(embeddings, model_iden, provider_model_iden, usage);

	if options_set.capture_raw_body() {
		response = response.with_captured_raw_body(body);
	}

	Ok(response)
}

// endregion: --- Public Functions
