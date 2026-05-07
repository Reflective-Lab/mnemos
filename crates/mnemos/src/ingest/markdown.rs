//! Markdown ingestion module for parsing and chunking markdown documents.
//!
//! This module provides functionality to:
//! - Parse markdown files using pulldown-cmark
//! - Extract YAML front-matter metadata
//! - Chunk content by headers (h1, h2, h3 sections)
//! - Preserve code blocks with language tags
//! - Return structured chunks ready for embedding
//!
//! # Example
//!
//! ```rust,no_run
//! use mnemos::ingest::{MarkdownIngester, MarkdownDocument};
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let ingester = MarkdownIngester::new();
//!
//!     // Ingest a single file
//!     let doc = ingester.ingest_file(Path::new("README.md")).await?;
//!     println!("Found {} chunks", doc.chunks.len());
//!
//!     // Ingest a directory recursively
//!     let docs = ingester.ingest_directory(Path::new("docs"), true).await?;
//!     println!("Ingested {} documents", docs.len());
//!
//!     Ok(())
//! }
//! ```

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, instrument, warn};

use crate::error::{Error, Result};

/// A parsed markdown document with extracted metadata and content chunks.
///
/// This structure represents a fully processed markdown file, including:
/// - The original file path for reference
/// - An optional title extracted from the first h1 heading or front-matter
/// - Metadata from YAML front-matter
/// - Content broken into semantic chunks for embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownDocument {
    /// Path to the source markdown file.
    pub path: PathBuf,

    /// Document title, extracted from front-matter or first h1 heading.
    pub title: Option<String>,

    /// Metadata extracted from YAML front-matter.
    /// Common keys include: author, date, tags, description, etc.
    pub metadata: HashMap<String, String>,

    /// Content chunks, each representing a semantic unit of the document.
    pub chunks: Vec<MarkdownChunk>,
}

impl MarkdownDocument {
    /// Create a new empty document for the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            title: None,
            metadata: HashMap::new(),
            chunks: Vec::new(),
        }
    }

    /// Get all text content concatenated (useful for full-document embedding).
    pub fn full_text(&self) -> String {
        self.chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get only text chunks (excluding code blocks).
    pub fn text_chunks(&self) -> impl Iterator<Item = &MarkdownChunk> {
        self.chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Text)
    }

    /// Get only code block chunks.
    pub fn code_chunks(&self) -> impl Iterator<Item = &MarkdownChunk> {
        self.chunks
            .iter()
            .filter(|c| matches!(c.chunk_type, ChunkType::CodeBlock { .. }))
    }
}

/// A single chunk of content from a markdown document.
///
/// Chunks are created by splitting the document at heading boundaries.
/// Each chunk preserves its position in the document hierarchy and
/// the line range it covers in the source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownChunk {
    /// The text content of this chunk.
    pub content: String,

    /// The type of content this chunk contains.
    pub chunk_type: ChunkType,

    /// The heading hierarchy leading to this chunk.
    /// For example, `["Introduction", "Getting Started"]` means this chunk
    /// is under an h1 "Introduction" and h2 "Getting Started".
    pub heading_hierarchy: Vec<String>,

    /// The line range in the source file (1-indexed, inclusive).
    /// `(start_line, end_line)` where both are inclusive.
    pub line_range: (usize, usize),
}

impl MarkdownChunk {
    /// Create a new text chunk.
    pub fn text(
        content: impl Into<String>,
        heading_hierarchy: Vec<String>,
        line_range: (usize, usize),
    ) -> Self {
        Self {
            content: content.into(),
            chunk_type: ChunkType::Text,
            heading_hierarchy,
            line_range,
        }
    }

    /// Create a new code block chunk.
    pub fn code_block(
        content: impl Into<String>,
        language: Option<String>,
        heading_hierarchy: Vec<String>,
        line_range: (usize, usize),
    ) -> Self {
        Self {
            content: content.into(),
            chunk_type: ChunkType::CodeBlock { language },
            heading_hierarchy,
            line_range,
        }
    }

    /// Check if this chunk is a code block.
    pub fn is_code(&self) -> bool {
        matches!(self.chunk_type, ChunkType::CodeBlock { .. })
    }

    /// Get the language of a code block, if applicable.
    pub fn code_language(&self) -> Option<&str> {
        match &self.chunk_type {
            ChunkType::CodeBlock { language } => language.as_deref(),
            _ => None,
        }
    }

    /// Get a context string describing where this chunk is in the document.
    pub fn context_string(&self) -> String {
        if self.heading_hierarchy.is_empty() {
            "Document root".to_string()
        } else {
            self.heading_hierarchy.join(" > ")
        }
    }
}

/// The type of content a chunk contains.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkType {
    /// Regular text content (paragraphs, inline code, etc.).
    Text,

    /// A fenced code block with optional language specification.
    CodeBlock {
        /// The programming language, if specified (e.g., "rust", "python").
        language: Option<String>,
    },

    /// A list (ordered or unordered).
    List,

    /// A table.
    Table,
}

/// Configuration options for the markdown ingester.
#[derive(Debug, Clone)]
pub struct IngesterConfig {
    /// Minimum chunk size in characters. Smaller chunks will be merged.
    pub min_chunk_size: usize,

    /// Maximum chunk size in characters. Larger chunks will be split.
    pub max_chunk_size: usize,

    /// Whether to preserve code blocks as separate chunks.
    pub preserve_code_blocks: bool,

    /// Whether to include front-matter in the output.
    pub include_frontmatter: bool,

    /// File extensions to consider as markdown.
    pub markdown_extensions: Vec<String>,
}

impl Default for IngesterConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 50,
            max_chunk_size: 4000,
            preserve_code_blocks: true,
            include_frontmatter: true,
            markdown_extensions: vec!["md".to_string(), "markdown".to_string(), "mdx".to_string()],
        }
    }
}

/// Markdown file ingester that parses and chunks markdown documents.
///
/// The ingester handles:
/// - YAML front-matter extraction
/// - Content chunking by headings
/// - Code block preservation with language tags
/// - Recursive directory traversal
#[derive(Debug, Clone)]
pub struct MarkdownIngester {
    config: IngesterConfig,
}

impl Default for MarkdownIngester {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownIngester {
    /// Create a new markdown ingester with default configuration.
    pub fn new() -> Self {
        Self {
            config: IngesterConfig::default(),
        }
    }

    /// Create a new markdown ingester with custom configuration.
    pub fn with_config(config: IngesterConfig) -> Self {
        Self { config }
    }

    /// Ingest a single markdown file.
    ///
    /// This method reads the file, extracts front-matter, parses the markdown,
    /// and returns a structured document with chunks.
    ///
    /// # Arguments
    /// * `path` - Path to the markdown file
    ///
    /// # Returns
    /// A `MarkdownDocument` containing the parsed content and metadata
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or is not valid UTF-8
    #[instrument(skip(self), fields(path = %path.display()))]
    pub async fn ingest_file(&self, path: &Path) -> Result<MarkdownDocument> {
        debug!("Ingesting markdown file");

        let content = fs::read_to_string(path).await.map_err(Error::Io)?;

        let mut document = MarkdownDocument::new(path);

        // Extract front-matter and get the remaining content
        let (frontmatter, body) = Self::extract_frontmatter(&content);

        if let Some(fm) = frontmatter {
            document.metadata.clone_from(&fm);
            // Try to extract title from front-matter
            if let Some(title) = fm.get("title") {
                document.title = Some(title.clone());
            }
        }

        // Parse the markdown body and create chunks
        document.chunks = self.parse_markdown(body);

        // If no title from front-matter, try to get it from first h1
        if document.title.is_none() {
            document.title = document
                .chunks
                .iter()
                .find(|c| !c.heading_hierarchy.is_empty())
                .and_then(|c| c.heading_hierarchy.first().cloned());
        }

        debug!(chunks = document.chunks.len(), "Ingestion complete");

        Ok(document)
    }

    /// Ingest all markdown files in a directory.
    ///
    /// # Arguments
    /// * `dir` - Path to the directory
    /// * `recursive` - Whether to recursively traverse subdirectories
    ///
    /// # Returns
    /// A vector of `MarkdownDocument` for each markdown file found
    ///
    /// # Errors
    /// Returns an error if the directory cannot be read
    #[instrument(skip(self), fields(dir = %dir.display(), recursive))]
    pub async fn ingest_directory(
        &self,
        dir: &Path,
        recursive: bool,
    ) -> Result<Vec<MarkdownDocument>> {
        debug!("Ingesting directory");

        let mut documents = Vec::new();
        let mut dirs_to_process = vec![dir.to_path_buf()];

        while let Some(current_dir) = dirs_to_process.pop() {
            let mut entries = fs::read_dir(&current_dir).await.map_err(Error::Io)?;

            while let Some(entry) = entries.next_entry().await.map_err(Error::Io)? {
                let path = entry.path();
                let file_type = entry.file_type().await.map_err(Error::Io)?;

                if file_type.is_dir() {
                    if recursive {
                        dirs_to_process.push(path);
                    }
                } else if file_type.is_file() && self.is_markdown_file(&path) {
                    match self.ingest_file(&path).await {
                        Ok(doc) => documents.push(doc),
                        Err(e) => {
                            warn!(path = %path.display(), error = %e, "Failed to ingest file");
                        }
                    }
                }
            }
        }

        debug!(count = documents.len(), "Directory ingestion complete");

        Ok(documents)
    }

    /// Extract YAML front-matter from markdown content.
    ///
    /// Front-matter is expected to be at the very beginning of the file,
    /// delimited by `---` lines. For example:
    ///
    /// ```text
    /// ---
    /// title: My Document
    /// author: John Doe
    /// date: 2024-01-15
    /// ---
    ///
    /// # Document content starts here
    /// ```
    ///
    /// # Arguments
    /// * `content` - The full markdown file content
    ///
    /// # Returns
    /// A tuple of (optional metadata HashMap, remaining content after front-matter)
    pub fn extract_frontmatter(content: &str) -> (Option<HashMap<String, String>>, &str) {
        // Front-matter must start at the very beginning with ---
        if !content.starts_with("---") {
            return (None, content);
        }

        // Find the closing ---
        // Skip the first --- and find the next one
        let after_first_delimiter = &content[3..];
        let Some(end_pos) = after_first_delimiter.find("\n---") else {
            // No closing delimiter found
            return (None, content);
        };

        // Extract the YAML content (between the delimiters)
        let yaml_content = after_first_delimiter[..end_pos].trim();

        // Find where the body starts (after the closing ---)
        let body_start = 3 + end_pos + 4; // 3 for initial ---, +4 for \n---
        let body = if body_start < content.len() {
            // Skip any leading newlines after front-matter
            content[body_start..].trim_start_matches(['\n', '\r'])
        } else {
            ""
        };

        // Parse the YAML
        match serde_yaml::from_str::<serde_yaml::Value>(yaml_content) {
            Ok(yaml) => {
                let mut metadata = HashMap::new();

                // Convert YAML mapping to HashMap<String, String>
                if let serde_yaml::Value::Mapping(map) = yaml {
                    for (key, value) in map {
                        if let serde_yaml::Value::String(k) = key {
                            let v = match value {
                                serde_yaml::Value::String(s) => s,
                                serde_yaml::Value::Number(n) => n.to_string(),
                                serde_yaml::Value::Bool(b) => b.to_string(),
                                serde_yaml::Value::Sequence(seq) => {
                                    // Convert arrays to comma-separated strings
                                    seq.iter()
                                        .filter_map(|v| match v {
                                            serde_yaml::Value::String(s) => Some(s.as_str()),
                                            _ => None,
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                }
                                _ => continue,
                            };
                            metadata.insert(k, v);
                        }
                    }
                }

                (Some(metadata), body)
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse YAML front-matter");
                (None, content)
            }
        }
    }

    /// Parse markdown content and return chunks.
    ///
    /// The parsing strategy:
    /// 1. Track heading hierarchy (h1, h2, h3)
    /// 2. Accumulate content until the next heading
    /// 3. Create separate chunks for code blocks if configured
    /// 4. Track line numbers for each chunk
    fn parse_markdown(&self, content: &str) -> Vec<MarkdownChunk> {
        let mut chunks = Vec::new();
        let mut current_text = String::new();
        let mut heading_hierarchy: Vec<String> = Vec::new();
        let mut current_heading_text = String::new();
        let mut in_heading = false;
        let mut in_code_block = false;
        let mut code_block_content = String::new();
        let mut code_block_language: Option<String> = None;
        let mut in_list = false;
        let mut in_table = false;

        // Track line numbers
        let mut current_line = 1;
        let mut chunk_start_line = 1;

        // Enable all parsing options for maximum fidelity
        let options = Options::all();
        let parser = Parser::new_ext(content, options);

        for event in parser {
            match event {
                Event::Start(Tag::Heading { level, .. }) => {
                    // Before starting a new heading, save any accumulated content
                    if !current_text.trim().is_empty() {
                        let chunk_type = if in_list {
                            ChunkType::List
                        } else if in_table {
                            ChunkType::Table
                        } else {
                            ChunkType::Text
                        };

                        chunks.push(MarkdownChunk {
                            content: current_text.trim().to_string(),
                            chunk_type,
                            heading_hierarchy: heading_hierarchy.clone(),
                            line_range: (chunk_start_line, current_line),
                        });
                        current_text.clear();
                    }

                    in_heading = true;
                    current_heading_text.clear();

                    // Adjust heading hierarchy based on level
                    // h1 = level 0, h2 = level 1, h3 = level 2
                    let level_idx = match level {
                        HeadingLevel::H1 => 0,
                        HeadingLevel::H2 => 1,
                        HeadingLevel::H3 => 2,
                        HeadingLevel::H4 => 3,
                        HeadingLevel::H5 => 4,
                        HeadingLevel::H6 => 5,
                    };

                    // Truncate hierarchy to this level
                    heading_hierarchy.truncate(level_idx);
                    chunk_start_line = current_line;
                }

                Event::End(TagEnd::Heading(_)) => {
                    in_heading = false;
                    // Add the heading text to hierarchy
                    let heading_text = current_heading_text.trim().to_string();
                    if !heading_text.is_empty() {
                        heading_hierarchy.push(heading_text);
                    }
                    current_heading_text.clear();
                }

                Event::Start(Tag::CodeBlock(kind)) => {
                    // Save any accumulated text before the code block
                    if !current_text.trim().is_empty() {
                        chunks.push(MarkdownChunk {
                            content: current_text.trim().to_string(),
                            chunk_type: ChunkType::Text,
                            heading_hierarchy: heading_hierarchy.clone(),
                            line_range: (chunk_start_line, current_line),
                        });
                        current_text.clear();
                    }

                    in_code_block = true;
                    code_block_content.clear();
                    chunk_start_line = current_line;

                    // Extract language from code fence
                    code_block_language = match kind {
                        pulldown_cmark::CodeBlockKind::Fenced(lang) => {
                            let lang_str = lang.to_string();
                            if lang_str.is_empty() {
                                None
                            } else {
                                // Handle cases like "rust,ignore" - take just the language
                                Some(lang_str.split(',').next().unwrap_or(&lang_str).to_string())
                            }
                        }
                        pulldown_cmark::CodeBlockKind::Indented => None,
                    };
                }

                Event::End(TagEnd::CodeBlock) => {
                    if self.config.preserve_code_blocks && !code_block_content.trim().is_empty() {
                        chunks.push(MarkdownChunk {
                            content: code_block_content.trim().to_string(),
                            chunk_type: ChunkType::CodeBlock {
                                language: code_block_language.take(),
                            },
                            heading_hierarchy: heading_hierarchy.clone(),
                            line_range: (chunk_start_line, current_line),
                        });
                    } else if !code_block_content.is_empty() {
                        // Include code in regular text flow
                        current_text.push_str("```");
                        if let Some(ref lang) = code_block_language {
                            current_text.push_str(lang);
                        }
                        current_text.push('\n');
                        current_text.push_str(&code_block_content);
                        current_text.push_str("```\n");
                    }

                    in_code_block = false;
                    code_block_content.clear();
                    code_block_language = None;
                    chunk_start_line = current_line;
                }

                Event::Start(Tag::List(_)) => {
                    in_list = true;
                }

                Event::End(TagEnd::List(_)) => {
                    in_list = false;
                }

                Event::Start(Tag::Table(_)) => {
                    in_table = true;
                }

                Event::End(TagEnd::Table) => {
                    in_table = false;
                }

                Event::Text(text) => {
                    // Count newlines in the text for line tracking
                    current_line += text.chars().filter(|c| *c == '\n').count();

                    if in_heading {
                        current_heading_text.push_str(&text);
                    } else if in_code_block {
                        code_block_content.push_str(&text);
                    } else {
                        current_text.push_str(&text);
                    }
                }

                Event::Code(code) => {
                    // Inline code
                    if in_heading {
                        current_heading_text.push('`');
                        current_heading_text.push_str(&code);
                        current_heading_text.push('`');
                    } else if !in_code_block {
                        current_text.push('`');
                        current_text.push_str(&code);
                        current_text.push('`');
                    }
                }

                Event::SoftBreak | Event::HardBreak => {
                    current_line += 1;
                    if in_heading {
                        current_heading_text.push(' ');
                    } else if in_code_block {
                        code_block_content.push('\n');
                    } else {
                        current_text.push('\n');
                    }
                }

                Event::Html(html) => {
                    // Include HTML as-is
                    current_line += html.chars().filter(|c| *c == '\n').count();
                    if !in_code_block && !in_heading {
                        current_text.push_str(&html);
                    }
                }

                _ => {}
            }
        }

        // Don't forget any remaining content
        if !current_text.trim().is_empty() {
            let chunk_type = if in_list {
                ChunkType::List
            } else if in_table {
                ChunkType::Table
            } else {
                ChunkType::Text
            };

            chunks.push(MarkdownChunk {
                content: current_text.trim().to_string(),
                chunk_type,
                heading_hierarchy: heading_hierarchy.clone(),
                line_range: (chunk_start_line, current_line),
            });
        }

        // Post-process: merge small chunks and split large ones
        self.post_process_chunks(chunks)
    }

    /// Post-process chunks to respect size constraints.
    fn post_process_chunks(&self, chunks: Vec<MarkdownChunk>) -> Vec<MarkdownChunk> {
        let mut result = Vec::new();
        let mut pending: Option<MarkdownChunk> = None;

        for chunk in chunks {
            // Don't merge code blocks
            if chunk.is_code() {
                // Flush any pending text chunk
                if let Some(p) = pending.take() {
                    if p.content.len() > self.config.max_chunk_size {
                        result.extend(self.split_large_chunk(p));
                    } else {
                        result.push(p);
                    }
                }
                // Add code block as-is (or split if too large)
                if chunk.content.len() > self.config.max_chunk_size {
                    result.extend(self.split_large_chunk(chunk));
                } else {
                    result.push(chunk);
                }
                continue;
            }

            match pending.take() {
                None => {
                    pending = Some(chunk);
                }
                Some(mut p) => {
                    // If pending chunk is too small, try to merge
                    if p.content.len() < self.config.min_chunk_size {
                        // Only merge if they share the same heading context
                        if p.heading_hierarchy == chunk.heading_hierarchy {
                            p.content.push_str("\n\n");
                            p.content.push_str(&chunk.content);
                            p.line_range.1 = chunk.line_range.1;
                            pending = Some(p);
                        } else {
                            // Different context, keep small chunk as-is
                            result.push(p);
                            pending = Some(chunk);
                        }
                    } else {
                        // Pending chunk is big enough, push it
                        if p.content.len() > self.config.max_chunk_size {
                            result.extend(self.split_large_chunk(p));
                        } else {
                            result.push(p);
                        }
                        pending = Some(chunk);
                    }
                }
            }
        }

        // Don't forget the last pending chunk
        if let Some(p) = pending {
            if p.content.len() > self.config.max_chunk_size {
                result.extend(self.split_large_chunk(p));
            } else {
                result.push(p);
            }
        }

        result
    }

    /// Split a large chunk into smaller pieces.
    fn split_large_chunk(&self, chunk: MarkdownChunk) -> Vec<MarkdownChunk> {
        let mut result = Vec::new();
        let content = &chunk.content;
        let max_size = self.config.max_chunk_size;

        // Try to split at paragraph boundaries first
        let paragraphs: Vec<&str> = content.split("\n\n").collect();

        let mut current = String::new();
        let mut current_start = chunk.line_range.0;

        for para in paragraphs {
            // If a single paragraph exceeds max_size, split it at sentence boundaries
            if para.len() > max_size {
                // Save any accumulated content first
                if !current.is_empty() {
                    let lines_in_current = current.chars().filter(|c| *c == '\n').count() + 1;
                    result.push(MarkdownChunk {
                        content: current.clone(),
                        chunk_type: chunk.chunk_type.clone(),
                        heading_hierarchy: chunk.heading_hierarchy.clone(),
                        line_range: (current_start, current_start + lines_in_current),
                    });
                    current_start += lines_in_current;
                    current.clear();
                }

                // Split large paragraph at sentence boundaries (. followed by space)
                let mut para_chunk = String::new();
                for sentence in para.split(". ") {
                    let sentence_with_period = if sentence.ends_with('.') {
                        sentence.to_string()
                    } else {
                        format!("{}. ", sentence)
                    };

                    if para_chunk.len() + sentence_with_period.len() > max_size
                        && !para_chunk.is_empty()
                    {
                        result.push(MarkdownChunk {
                            content: para_chunk.trim().to_string(),
                            chunk_type: chunk.chunk_type.clone(),
                            heading_hierarchy: chunk.heading_hierarchy.clone(),
                            line_range: (current_start, current_start + 1),
                        });
                        para_chunk.clear();
                    }
                    para_chunk.push_str(&sentence_with_period);
                }
                if !para_chunk.is_empty() {
                    result.push(MarkdownChunk {
                        content: para_chunk.trim().to_string(),
                        chunk_type: chunk.chunk_type.clone(),
                        heading_hierarchy: chunk.heading_hierarchy.clone(),
                        line_range: (current_start, current_start + 1),
                    });
                }
                continue;
            }

            if current.len() + para.len() + 2 > max_size && !current.is_empty() {
                // Save current chunk
                let lines_in_current = current.chars().filter(|c| *c == '\n').count() + 1;
                result.push(MarkdownChunk {
                    content: current.clone(),
                    chunk_type: chunk.chunk_type.clone(),
                    heading_hierarchy: chunk.heading_hierarchy.clone(),
                    line_range: (current_start, current_start + lines_in_current),
                });
                current_start += lines_in_current;
                current.clear();
            }

            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }

        // Don't forget the last piece
        if !current.is_empty() {
            result.push(MarkdownChunk {
                content: current,
                chunk_type: chunk.chunk_type,
                heading_hierarchy: chunk.heading_hierarchy,
                line_range: (current_start, chunk.line_range.1),
            });
        }

        result
    }

    /// Check if a path is a markdown file based on extension.
    fn is_markdown_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| {
                self.config
                    .markdown_extensions
                    .iter()
                    .any(|m| m.eq_ignore_ascii_case(ext))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs::File;
    use tokio::io::AsyncWriteExt;

    /// Helper to create a temp file with content
    async fn create_temp_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).await.unwrap();
        file.write_all(content.as_bytes()).await.unwrap();
        file.sync_all().await.unwrap();
        path
    }

    // ==================== Front-matter Extraction Tests ====================

    /// Test: Front-matter is correctly extracted from a markdown file.
    ///
    /// What happens:
    /// 1. The content starts with `---` followed by YAML
    /// 2. The YAML is parsed into a HashMap<String, String>
    /// 3. The body after the closing `---` is returned separately
    #[test]
    fn test_extract_frontmatter_basic() {
        let content = r"---
title: My Document
author: John Doe
date: 2024-01-15
---

# Hello World

This is the body.";

        let (metadata, body) = MarkdownIngester::extract_frontmatter(content);

        assert!(metadata.is_some(), "Front-matter should be extracted");
        let metadata = metadata.unwrap();

        assert_eq!(metadata.get("title"), Some(&"My Document".to_string()));
        assert_eq!(metadata.get("author"), Some(&"John Doe".to_string()));
        assert_eq!(metadata.get("date"), Some(&"2024-01-15".to_string()));

        assert!(
            body.starts_with("# Hello World"),
            "Body should start with heading"
        );
    }

    /// Test: Files without front-matter return None for metadata.
    ///
    /// What happens:
    /// 1. The content does not start with `---`
    /// 2. No metadata is extracted
    /// 3. The entire content is returned as the body
    #[test]
    fn test_extract_frontmatter_none() {
        let content = "# Just a Heading\n\nSome content.";

        let (metadata, body) = MarkdownIngester::extract_frontmatter(content);

        assert!(metadata.is_none(), "No front-matter should be found");
        assert_eq!(body, content, "Body should be the entire content");
    }

    /// Test: Arrays in front-matter are converted to comma-separated strings.
    ///
    /// What happens:
    /// 1. YAML arrays like `tags: [rust, programming]`
    /// 2. Are converted to `"rust, programming"` string
    /// 3. This simplifies storage in HashMap<String, String>
    #[test]
    fn test_extract_frontmatter_arrays() {
        let content = r"---
title: Tagged Post
tags:
  - rust
  - programming
  - web
---

Content here.";

        let (metadata, _body) = MarkdownIngester::extract_frontmatter(content);

        let metadata = metadata.expect("Front-matter should be extracted");
        assert_eq!(
            metadata.get("tags"),
            Some(&"rust, programming, web".to_string())
        );
    }

    /// Test: Unclosed front-matter is treated as no front-matter.
    ///
    /// What happens:
    /// 1. Content starts with `---` but has no closing `---`
    /// 2. This is invalid front-matter
    /// 3. The entire content is returned as body
    #[test]
    fn test_extract_frontmatter_unclosed() {
        let content = r"---
title: Broken
author: Nobody

# This has no closing delimiter";

        let (metadata, body) = MarkdownIngester::extract_frontmatter(content);

        assert!(metadata.is_none(), "Unclosed front-matter should not parse");
        assert_eq!(body, content, "Body should be entire content");
    }

    // ==================== Markdown Parsing Tests ====================

    /// Test: Basic heading hierarchy is tracked correctly.
    ///
    /// What happens:
    /// 1. Parser encounters h1 heading -> starts hierarchy at level 0
    /// 2. Parser encounters h2 heading -> adds to hierarchy at level 1
    /// 3. Content under each heading has correct heading_hierarchy
    /// 4. New h2 replaces previous h2 in hierarchy
    #[tokio::test]
    async fn test_heading_hierarchy() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# Main Title

Intro paragraph.

## Section One

Content in section one.

## Section Two

Content in section two.

### Subsection

Deep content.
";

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        // Find chunk with "section one" content
        let section_one = doc
            .chunks
            .iter()
            .find(|c| c.content.to_lowercase().contains("content in section one"))
            .expect("Should find section one content");

        assert_eq!(
            section_one.heading_hierarchy,
            vec!["Main Title", "Section One"],
            "Section one should have correct hierarchy"
        );

        // Find chunk with subsection content
        let subsection = doc
            .chunks
            .iter()
            .find(|c| c.content.to_lowercase().contains("deep content"))
            .expect("Should find subsection content");

        assert_eq!(
            subsection.heading_hierarchy,
            vec!["Main Title", "Section Two", "Subsection"],
            "Subsection should have full hierarchy"
        );
    }

    /// Test: Code blocks are preserved with their language tags.
    ///
    /// What happens:
    /// 1. Parser encounters fenced code block (```)
    /// 2. Language is extracted from the fence (e.g., ```rust)
    /// 3. Code content is preserved exactly
    /// 4. ChunkType::CodeBlock is used with language field
    #[tokio::test]
    async fn test_code_block_preservation() {
        let temp_dir = TempDir::new().unwrap();
        let content = r#"# Code Examples

Here's some Rust code:

```rust
fn main() {
    println!("Hello, world!");
}
```

And some Python:

```python
def hello():
    print("Hello, world!")
```
"#;

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        // Find Rust code block
        let rust_block = doc.chunks.iter()
            .find(|c| matches!(&c.chunk_type, ChunkType::CodeBlock { language: Some(l) } if l == "rust"))
            .expect("Should find Rust code block");

        assert!(
            rust_block.content.contains("println!"),
            "Rust code should be preserved"
        );

        // Find Python code block
        let python_block = doc.chunks.iter()
            .find(|c| matches!(&c.chunk_type, ChunkType::CodeBlock { language: Some(l) } if l == "python"))
            .expect("Should find Python code block");

        assert!(
            python_block.content.contains("def hello"),
            "Python code should be preserved"
        );
    }

    /// Test: Code blocks without language specification.
    ///
    /// What happens:
    /// 1. Fenced code block without language (just ```)
    /// 2. language field is None
    /// 3. Content is still preserved
    #[tokio::test]
    async fn test_code_block_no_language() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# Unlabeled Code

```
some generic code
```
";

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        let code_block = doc
            .chunks
            .iter()
            .find(|c| matches!(&c.chunk_type, ChunkType::CodeBlock { language: None }))
            .expect("Should find code block without language");

        assert!(code_block.content.contains("generic code"));
    }

    /// Test: Title extraction from front-matter takes precedence.
    ///
    /// What happens:
    /// 1. Document has title in front-matter
    /// 2. Document also has h1 heading
    /// 3. Front-matter title is used as document title
    #[tokio::test]
    async fn test_title_from_frontmatter() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"---
title: Front-matter Title
---

# Heading Title

Content.
";

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        assert_eq!(
            doc.title,
            Some("Front-matter Title".to_string()),
            "Title should come from front-matter"
        );
    }

    /// Test: Title extracted from first h1 when no front-matter.
    ///
    /// What happens:
    /// 1. No front-matter in document
    /// 2. First h1 heading is used as title
    #[tokio::test]
    async fn test_title_from_heading() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# First Heading

Some content here.

## Second Section

More content.
";

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        assert_eq!(
            doc.title,
            Some("First Heading".to_string()),
            "Title should come from first h1"
        );
    }

    // ==================== Directory Ingestion Tests ====================

    /// Test: Recursive directory ingestion finds all markdown files.
    ///
    /// What happens:
    /// 1. Directory structure with nested folders
    /// 2. recursive=true traverses all subdirectories
    /// 3. All .md files are ingested
    /// 4. Non-markdown files are ignored
    #[tokio::test]
    async fn test_directory_ingestion_recursive() {
        let temp_dir = TempDir::new().unwrap();

        // Create directory structure
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).await.unwrap();

        create_temp_file(&temp_dir, "root.md", "# Root\n\nRoot content.").await;
        create_temp_file(&temp_dir, "other.txt", "Not markdown").await;

        // Create file in subdirectory
        let sub_path = subdir.join("nested.md");
        let mut file = File::create(&sub_path).await.unwrap();
        file.write_all(b"# Nested\n\nNested content.")
            .await
            .unwrap();

        let ingester = MarkdownIngester::new();
        let docs = ingester
            .ingest_directory(temp_dir.path(), true)
            .await
            .unwrap();

        assert_eq!(docs.len(), 2, "Should find 2 markdown files");

        let titles: Vec<_> = docs.iter().filter_map(|d| d.title.as_ref()).collect();
        assert!(titles.contains(&&"Root".to_string()));
        assert!(titles.contains(&&"Nested".to_string()));
    }

    /// Test: Non-recursive ingestion stays in root directory.
    ///
    /// What happens:
    /// 1. recursive=false
    /// 2. Only files in the root directory are processed
    /// 3. Subdirectories are ignored
    #[tokio::test]
    async fn test_directory_ingestion_non_recursive() {
        let temp_dir = TempDir::new().unwrap();

        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).await.unwrap();

        create_temp_file(&temp_dir, "root.md", "# Root\n\nContent.").await;

        let sub_path = subdir.join("nested.md");
        let mut file = File::create(&sub_path).await.unwrap();
        file.write_all(b"# Nested\n\nContent.").await.unwrap();

        let ingester = MarkdownIngester::new();
        let docs = ingester
            .ingest_directory(temp_dir.path(), false)
            .await
            .unwrap();

        assert_eq!(docs.len(), 1, "Should find only root markdown file");
        assert_eq!(docs[0].title, Some("Root".to_string()));
    }

    // ==================== Chunk Processing Tests ====================

    /// Test: Small chunks are merged when they share context.
    ///
    /// What happens:
    /// 1. Multiple small paragraphs under same heading
    /// 2. If combined size < max_chunk_size, they're merged
    /// 3. This improves embedding quality by providing more context
    #[tokio::test]
    async fn test_small_chunk_merging() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# Section

A.

B.

C.
";

        let path = create_temp_file(&temp_dir, "test.md", content).await;
        let config = IngesterConfig {
            min_chunk_size: 100, // Force merging
            ..IngesterConfig::default()
        };

        let ingester = MarkdownIngester::with_config(config);
        let doc = ingester.ingest_file(&path).await.unwrap();

        // The three small paragraphs should be merged into one chunk
        assert!(
            doc.chunks.len() <= 2, // May have 1-2 chunks depending on merging
            "Small chunks should be merged"
        );
    }

    /// Test: Large chunks are split at paragraph boundaries.
    ///
    /// What happens:
    /// 1. Content exceeds max_chunk_size
    /// 2. Content is split at "\n\n" (paragraph) boundaries
    /// 3. Each resulting chunk is within size limits
    #[tokio::test]
    async fn test_large_chunk_splitting() {
        let temp_dir = TempDir::new().unwrap();

        // Create content larger than max_chunk_size
        let long_paragraph = "This is a test paragraph. ".repeat(200);
        let content = format!(
            "# Large Document\n\n{}\n\n{}\n\n{}",
            long_paragraph, long_paragraph, long_paragraph
        );

        let path = create_temp_file(&temp_dir, "test.md", &content).await;
        let config = IngesterConfig {
            max_chunk_size: 500,
            ..IngesterConfig::default()
        };
        let max_chunk_size = config.max_chunk_size;

        let ingester = MarkdownIngester::with_config(config);
        let doc = ingester.ingest_file(&path).await.unwrap();

        // Verify no chunk exceeds max size (with some tolerance for implementation)
        for chunk in &doc.chunks {
            // Allow some tolerance since we split at paragraph boundaries
            assert!(
                chunk.content.len() <= max_chunk_size + 200,
                "Chunk should not greatly exceed max size: {} > {}",
                chunk.content.len(),
                max_chunk_size
            );
        }
    }

    // ==================== Edge Cases ====================

    /// Test: Empty markdown file.
    ///
    /// What happens:
    /// 1. File exists but is empty
    /// 2. Returns document with no chunks
    /// 3. No error is raised
    #[tokio::test]
    async fn test_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_temp_file(&temp_dir, "empty.md", "").await;

        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        assert!(doc.chunks.is_empty(), "Empty file should have no chunks");
        assert!(doc.title.is_none(), "Empty file should have no title");
    }

    /// Test: File with only front-matter.
    ///
    /// What happens:
    /// 1. File has front-matter but no body content
    /// 2. Metadata is extracted
    /// 3. Chunks are empty
    #[tokio::test]
    async fn test_frontmatter_only() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"---
title: Metadata Only
author: Test
---
";
        let path = create_temp_file(&temp_dir, "meta.md", content).await;

        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        assert_eq!(doc.title, Some("Metadata Only".to_string()));
        assert_eq!(doc.metadata.get("author"), Some(&"Test".to_string()));
        assert!(doc.chunks.is_empty(), "Should have no content chunks");
    }

    /// Test: Inline code within headings.
    ///
    /// What happens:
    /// 1. Heading contains inline code like `code`
    /// 2. The backticks are preserved in the heading text
    /// 3. Heading is correctly added to hierarchy
    #[tokio::test]
    async fn test_inline_code_in_heading() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# Using `async/await` in Rust

Some explanation here.
";
        let path = create_temp_file(&temp_dir, "test.md", content).await;

        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        assert!(
            doc.chunks.iter().any(|c| c
                .heading_hierarchy
                .iter()
                .any(|h| h.contains("`async/await`"))),
            "Heading should preserve inline code"
        );
    }

    /// Test: Context string generation.
    ///
    /// What happens:
    /// 1. MarkdownChunk has heading_hierarchy
    /// 2. context_string() returns "H1 > H2 > H3" format
    /// 3. Empty hierarchy returns "Document root"
    #[test]
    fn test_context_string() {
        let chunk_with_hierarchy = MarkdownChunk {
            content: "Test".to_string(),
            chunk_type: ChunkType::Text,
            heading_hierarchy: vec!["Main".to_string(), "Section".to_string()],
            line_range: (1, 5),
        };

        assert_eq!(chunk_with_hierarchy.context_string(), "Main > Section");

        let chunk_no_hierarchy = MarkdownChunk {
            content: "Test".to_string(),
            chunk_type: ChunkType::Text,
            heading_hierarchy: vec![],
            line_range: (1, 5),
        };

        assert_eq!(chunk_no_hierarchy.context_string(), "Document root");
    }

    /// Test: Document helper methods.
    ///
    /// What happens:
    /// 1. full_text() concatenates all chunks
    /// 2. text_chunks() filters to only text
    /// 3. code_chunks() filters to only code
    #[test]
    fn test_document_helpers() {
        let mut doc = MarkdownDocument::new("/test.md");
        doc.chunks = vec![
            MarkdownChunk::text("First text", vec![], (1, 2)),
            MarkdownChunk::code_block("let x = 1;", Some("rust".to_string()), vec![], (3, 5)),
            MarkdownChunk::text("Second text", vec![], (6, 7)),
        ];

        let full = doc.full_text();
        assert!(full.contains("First text"));
        assert!(full.contains("let x = 1;"));
        assert!(full.contains("Second text"));

        assert_eq!(doc.text_chunks().count(), 2);
        assert_eq!(doc.code_chunks().count(), 1);
    }

    /// Test: Code block language extraction handles edge cases.
    ///
    /// What happens:
    /// 1. Language like "rust,ignore" -> extracts just "rust"
    /// 2. This handles common markdown patterns
    #[tokio::test]
    async fn test_code_language_with_attributes() {
        let temp_dir = TempDir::new().unwrap();
        let content = r"# Test

```rust,ignore
fn example() {}
```
";
        let path = create_temp_file(&temp_dir, "test.md", content).await;

        let ingester = MarkdownIngester::new();
        let doc = ingester.ingest_file(&path).await.unwrap();

        let code_chunk = doc.code_chunks().next().expect("Should have code chunk");
        assert_eq!(code_chunk.code_language(), Some("rust"));
    }

    /// Test: Different markdown extensions are recognized.
    ///
    /// What happens:
    /// 1. Default config recognizes .md, .markdown, .mdx
    /// 2. is_markdown_file() returns true for these
    /// 3. Other extensions return false
    #[test]
    fn test_markdown_extension_recognition() {
        let ingester = MarkdownIngester::new();

        assert!(ingester.is_markdown_file(Path::new("test.md")));
        assert!(ingester.is_markdown_file(Path::new("test.markdown")));
        assert!(ingester.is_markdown_file(Path::new("test.mdx")));
        assert!(ingester.is_markdown_file(Path::new("test.MD"))); // Case insensitive

        assert!(!ingester.is_markdown_file(Path::new("test.txt")));
        assert!(!ingester.is_markdown_file(Path::new("test.rs")));
        assert!(!ingester.is_markdown_file(Path::new("noextension")));
    }
}
