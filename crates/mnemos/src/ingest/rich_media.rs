//! Shared rich-media ingestion and transcription contracts.
//!
//! These abstractions support Phase 3 work (audio processing, video
//! transcription) without coupling the rest of the ingest pipeline to a
//! specific transcription backend or runtime.

use crate::Result;
use crate::error::Error;
use crate::ingest::SourceProvenance;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// The media type being ingested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    /// Audio-only media.
    Audio,
    /// Video media (may also include audio track).
    Video,
}

/// Transcription backend family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionEngine {
    /// OpenAI Whisper or compatible local implementation.
    Whisper,
    /// Test/mock backend.
    Mock,
    /// Other external backend.
    External,
}

/// Request for media ingestion or preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaIngestRequest {
    /// Path to the media file.
    pub path: PathBuf,
    /// Whether this is audio or video.
    pub media_kind: MediaKind,
    /// Shared source provenance metadata.
    pub provenance: SourceProvenance,
    /// Optional language hints.
    pub language_hints: Vec<String>,
    /// Source-specific metadata (codec, channels, sample-rate, etc.).
    pub metadata: HashMap<String, String>,
}

impl MediaIngestRequest {
    /// Create a new media ingest request.
    pub fn new(
        path: impl Into<PathBuf>,
        media_kind: MediaKind,
        provenance: SourceProvenance,
    ) -> Self {
        Self {
            path: path.into(),
            media_kind,
            provenance,
            language_hints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a language hint.
    pub fn with_language_hint(mut self, language: impl Into<String>) -> Self {
        self.language_hints.push(language.into());
        self
    }
}

/// Chunking policy used when segmenting transcript text for indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptChunkPolicy {
    /// Maximum transcript segment duration in milliseconds.
    pub max_segment_duration_ms: u64,
    /// Maximum characters per chunk.
    pub max_chars_per_chunk: usize,
}

impl Default for TranscriptChunkPolicy {
    fn default() -> Self {
        Self {
            max_segment_duration_ms: 60_000,
            max_chars_per_chunk: 2_000,
        }
    }
}

/// Request passed to transcription backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    /// Base media request.
    pub media: MediaIngestRequest,
    /// Transcript chunking hint for downstream ingestion.
    pub chunk_policy: TranscriptChunkPolicy,
    /// Whether speaker labels are desired (if backend supports them).
    pub diarization: bool,
}

impl TranscriptionRequest {
    /// Create a transcription request from a media request.
    pub fn new(media: MediaIngestRequest) -> Self {
        Self {
            media,
            chunk_policy: TranscriptChunkPolicy::default(),
            diarization: false,
        }
    }

    /// Enable speaker diarization if supported.
    pub fn with_diarization(mut self) -> Self {
        self.diarization = true;
        self
    }
}

/// Timestamped transcript segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// Segment index in transcript order.
    pub index: usize,
    /// Start timestamp in milliseconds from media start.
    pub start_ms: u64,
    /// End timestamp in milliseconds from media start.
    pub end_ms: u64,
    /// Transcript text for this segment.
    pub text: String,
    /// Optional speaker identifier.
    pub speaker: Option<String>,
    /// Confidence score when provided by backend.
    pub confidence: Option<f32>,
}

impl TranscriptSegment {
    /// Create a segment.
    pub fn new(index: usize, start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            index,
            start_ms,
            end_ms,
            text: text.into(),
            speaker: None,
            confidence: None,
        }
    }

    /// Segment duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if timestamps are a valid non-empty span.
    pub fn has_valid_time_range(&self) -> bool {
        self.end_ms > self.start_ms
    }
}

/// A full transcript result for audio or video media.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptDocument {
    /// Backend engine used to produce the transcript.
    pub engine: TranscriptionEngine,
    /// Media type for this transcript.
    pub media_kind: MediaKind,
    /// Optional detected language code.
    pub language: Option<String>,
    /// Transcript segments in chronological order.
    pub segments: Vec<TranscriptSegment>,
    /// Optional backend-provided full transcript text.
    pub full_text: String,
    /// When the transcript was produced.
    pub generated_at: DateTime<Utc>,
    /// Shared source provenance metadata.
    pub provenance: SourceProvenance,
}

impl TranscriptDocument {
    /// Create an empty transcript document.
    pub fn new(
        engine: TranscriptionEngine,
        media_kind: MediaKind,
        provenance: SourceProvenance,
    ) -> Self {
        Self {
            engine,
            media_kind,
            language: None,
            segments: Vec::new(),
            full_text: String::new(),
            generated_at: Utc::now(),
            provenance,
        }
    }

    /// Returns the best available transcript text for indexing.
    pub fn effective_text(&self) -> String {
        let trimmed = self.full_text.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }

        self.segments
            .iter()
            .map(|segment| segment.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Validate transcript segment ordering and timestamp ranges.
    pub fn validate(&self) -> Result<()> {
        let mut last_end = 0u64;

        for (position, segment) in self.segments.iter().enumerate() {
            if !segment.has_valid_time_range() {
                return Err(Error::ingest(format!(
                    "invalid transcript segment at position {position}: end_ms ({}) must be greater than start_ms ({})",
                    segment.end_ms, segment.start_ms
                )));
            }

            if position > 0 && segment.start_ms < last_end {
                return Err(Error::ingest(format!(
                    "overlapping transcript segment at position {position}: start_ms ({}) < previous end_ms ({last_end})",
                    segment.start_ms
                )));
            }

            last_end = segment.end_ms;
        }

        Ok(())
    }
}

/// Transcription backend contract used by audio/video ingestion pipelines.
#[async_trait]
pub trait TranscriptionBackend: Send + Sync {
    /// Backend engine identifier.
    fn engine(&self) -> TranscriptionEngine;

    /// Produce a transcript for the requested media.
    async fn transcribe(&self, request: &TranscriptionRequest) -> Result<TranscriptDocument>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{SourceKind, SourceProvenance};

    #[test]
    fn transcript_effective_text_falls_back_to_segments() {
        let provenance = SourceProvenance::new(SourceKind::Audio, "file:///meeting.m4a");
        let mut doc =
            TranscriptDocument::new(TranscriptionEngine::Mock, MediaKind::Audio, provenance);
        doc.segments
            .push(TranscriptSegment::new(0, 0, 1000, "hello world"));
        doc.segments
            .push(TranscriptSegment::new(1, 1000, 2000, "second segment"));

        assert_eq!(doc.effective_text(), "hello world\nsecond segment");
    }

    #[test]
    fn transcript_validation_rejects_overlap() {
        let provenance = SourceProvenance::new(SourceKind::Video, "file:///clip.mp4");
        let mut doc =
            TranscriptDocument::new(TranscriptionEngine::Mock, MediaKind::Video, provenance);
        doc.segments.push(TranscriptSegment::new(0, 0, 1500, "a"));
        doc.segments
            .push(TranscriptSegment::new(1, 1000, 2000, "b"));

        let err = doc.validate().unwrap_err();
        assert!(err.to_string().contains("overlapping transcript segment"));
    }

    #[test]
    fn transcript_validation_accepts_monotonic_segments() {
        let provenance = SourceProvenance::new(SourceKind::Video, "file:///clip.mp4");
        let mut doc =
            TranscriptDocument::new(TranscriptionEngine::Mock, MediaKind::Video, provenance);
        doc.segments.push(TranscriptSegment::new(0, 0, 1500, "a"));
        doc.segments
            .push(TranscriptSegment::new(1, 1500, 2200, "b"));

        assert!(doc.validate().is_ok());
    }
}
