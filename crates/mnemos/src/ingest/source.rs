//! Shared source provenance contracts for ingestion pipelines.
//!
//! These types are intentionally backend-agnostic and are used by upcoming
//! Phase 2 (Apple ecosystem) and Phase 3 (rich media) ingestion paths.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// The high-level source type an ingested artifact came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Apple Notes exported content.
    AppleNote,
    /// Screenshot image files.
    Screenshot,
    /// Photo image files.
    Photo,
    /// Video media files.
    Video,
    /// Audio media files.
    Audio,
    /// PDF documents.
    Pdf,
    /// Markdown documents.
    Markdown,
    /// Any other or not-yet-classified source.
    Unknown,
}

impl SourceKind {
    /// String form used in metadata keys and idempotency keys.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppleNote => "apple_note",
            Self::Screenshot => "screenshot",
            Self::Photo => "photo",
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Pdf => "pdf",
            Self::Markdown => "markdown",
            Self::Unknown => "unknown",
        }
    }
}

/// Provenance metadata shared by all ingesters.
///
/// This supports consistent source attribution and deterministic re-ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceProvenance {
    /// What kind of source produced the content.
    pub source_kind: SourceKind,
    /// Stable source URI/path (e.g., `file:///...`, `notes://...`).
    pub source_uri: String,
    /// Optional upstream source identifier (e.g., Apple Note ID, Photos asset ID).
    pub origin_id: Option<String>,
    /// Optional content fingerprint (checksum, hash, opaque digest).
    pub fingerprint: Option<String>,
    /// When the source artifact was originally created/captured, if known.
    pub captured_at: Option<DateTime<Utc>>,
    /// When this provenance record was created during import.
    pub imported_at: DateTime<Utc>,
    /// Arbitrary source-specific metadata.
    pub metadata: HashMap<String, String>,
}

impl SourceProvenance {
    /// Create a new provenance record.
    pub fn new(source_kind: SourceKind, source_uri: impl Into<String>) -> Self {
        Self {
            source_kind,
            source_uri: source_uri.into(),
            origin_id: None,
            fingerprint: None,
            captured_at: None,
            imported_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set the upstream origin ID.
    pub fn with_origin_id(mut self, origin_id: impl Into<String>) -> Self {
        self.origin_id = Some(origin_id.into());
        self
    }

    /// Set the source fingerprint.
    pub fn with_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.fingerprint = Some(fingerprint.into());
        self
    }

    /// Set the capture timestamp.
    pub fn with_captured_at(mut self, captured_at: DateTime<Utc>) -> Self {
        self.captured_at = Some(captured_at);
        self
    }

    /// Add a source-specific metadata key/value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Generate a deterministic idempotency key for de-duplication.
    ///
    /// Preference order:
    /// 1. `origin_id` (best external identity)
    /// 2. `fingerprint` (content identity)
    /// 3. `source_uri` (path identity fallback)
    pub fn idempotency_key(&self) -> String {
        if let Some(origin_id) = &self.origin_id {
            return format!("{}:origin:{}", self.source_kind.as_str(), origin_id);
        }

        if let Some(fingerprint) = &self.fingerprint {
            return format!("{}:fingerprint:{}", self.source_kind.as_str(), fingerprint);
        }

        format!("{}:uri:{}", self.source_kind.as_str(), self.source_uri)
    }

    /// Export normalized metadata pairs for attaching to knowledge entries.
    ///
    /// Keys are namespaced under `source.*`.
    pub fn metadata_pairs(&self) -> Vec<(String, String)> {
        let mut out = BTreeMap::new();
        out.insert(
            "source.kind".to_string(),
            self.source_kind.as_str().to_string(),
        );
        out.insert("source.uri".to_string(), self.source_uri.clone());
        out.insert("source.idempotency_key".to_string(), self.idempotency_key());

        if let Some(origin_id) = &self.origin_id {
            out.insert("source.origin_id".to_string(), origin_id.clone());
        }
        if let Some(fingerprint) = &self.fingerprint {
            out.insert("source.fingerprint".to_string(), fingerprint.clone());
        }
        if let Some(captured_at) = self.captured_at {
            out.insert("source.captured_at".to_string(), captured_at.to_rfc3339());
        }
        out.insert(
            "source.imported_at".to_string(),
            self.imported_at.to_rfc3339(),
        );

        for (key, value) in &self.metadata {
            out.insert(format!("source.meta.{key}"), value.clone());
        }

        out.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn idempotency_key_prefers_origin_id() {
        let provenance = SourceProvenance::new(SourceKind::AppleNote, "notes://abc")
            .with_fingerprint("sha256:deadbeef")
            .with_origin_id("note-123");

        assert_eq!(
            provenance.idempotency_key(),
            "apple_note:origin:note-123".to_string()
        );
    }

    #[test]
    fn metadata_pairs_are_namespaced_and_sorted() {
        let captured = Utc.with_ymd_and_hms(2025, 1, 2, 3, 4, 5).unwrap();
        let mut provenance = SourceProvenance::new(SourceKind::Screenshot, "/tmp/shot.png")
            .with_fingerprint("abc123")
            .with_captured_at(captured);
        provenance.imported_at = Utc.with_ymd_and_hms(2025, 1, 2, 10, 11, 12).unwrap();
        provenance
            .metadata
            .insert("window_title".into(), "Mail".into());
        provenance.metadata.insert("app".into(), "Mail".into());

        let pairs = provenance.metadata_pairs();
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        assert!(keys.contains(&"source.kind"));
        assert!(keys.contains(&"source.meta.window_title"));
        assert!(keys.contains(&"source.idempotency_key"));
    }
}
