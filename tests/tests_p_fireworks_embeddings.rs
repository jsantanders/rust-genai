mod support;

use crate::support::{TestResult, common_tests};
use genai::Client;
use genai::embed::EmbedOptions;

const MODEL_NS: &str = "fireworks::qwen3-embedding-8b";
const MODEL_RAW: &str = "fireworks/qwen3-embedding-8b";

// region:    --- Single Embedding Tests

#[tokio::test]
async fn test_fireworks_embed_single_simple_ok() -> TestResult<()> {
	common_tests::common_test_embed_single_simple_ok(MODEL_NS).await
}

#[tokio::test]
async fn test_fireworks_embed_single_raw_model_name_ok() -> TestResult<()> {
	common_tests::common_test_embed_single_simple_ok(MODEL_RAW).await
}

// endregion: --- Single Embedding Tests

// region:    --- Batch Embedding Tests

#[tokio::test]
async fn test_fireworks_embed_batch_simple_ok() -> TestResult<()> {
	common_tests::common_test_embed_batch_simple_ok(MODEL_NS).await
}

// endregion: --- Batch Embedding Tests

// region:    --- Options Tests

#[tokio::test]
async fn test_fireworks_embed_with_dimensions_ok() -> TestResult<()> {
	let client = Client::default();
	let options = EmbedOptions::new().with_dimensions(512).with_capture_usage(true);

	let response = client.embed(MODEL_NS, "Fireworks dimensions test", Some(&options)).await?;

	let embedding = response.first_embedding().ok_or("Should have embedding")?;
	assert_eq!(embedding.dimensions(), 512);
	assert_eq!(embedding.vector().len(), 512);
	assert!(response.usage.prompt_tokens.is_some());

	Ok(())
}

// endregion: --- Options Tests
