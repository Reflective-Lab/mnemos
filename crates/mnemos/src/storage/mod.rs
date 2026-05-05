//! Storage backend for persistence.

use crate::core::KnowledgeEntry;
use crate::error::{Error, Result};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Storage backend for persisting knowledge entries and embeddings.
pub struct StorageBackend {
    /// Path to storage directory.
    path: PathBuf,

    /// In-memory index.
    index: RwLock<StorageIndex>,

    /// Dirty flag for pending writes.
    dirty: RwLock<bool>,
}

/// Index of stored entries.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct StorageIndex {
    entries: HashMap<Uuid, EntryMetadata>,
    version: u32,
}

/// Metadata for a stored entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntryMetadata {
    id: Uuid,
    title: String,
    file_offset: u64,
}

impl StorageBackend {
    /// Open or create storage at the given path.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Load or create index
        let index = Self::load_index(&path).await.unwrap_or_default();

        Ok(Self {
            path,
            index: RwLock::new(index),
            dirty: RwLock::new(false),
        })
    }

    /// Load index from disk.
    async fn load_index(path: &Path) -> Result<StorageIndex> {
        let index_path = Self::index_path(path);

        if !index_path.exists() {
            return Ok(StorageIndex::default());
        }

        let mut file = fs::File::open(&index_path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        bincode::deserialize(&data).map_err(|e| Error::storage(e.to_string()))
    }

    /// Get index file path.
    fn index_path(base: &Path) -> PathBuf {
        base.with_extension("index")
    }

    /// Get data file path.
    fn data_path(base: &Path) -> PathBuf {
        base.with_extension("data")
    }

    /// Get embeddings file path.
    fn embeddings_path(base: &Path) -> PathBuf {
        base.with_extension("embeddings")
    }

    /// Load all entries and embeddings.
    pub async fn load_all(&self) -> Result<Vec<(KnowledgeEntry, Vec<f32>)>> {
        let data_path = Self::data_path(&self.path);
        let embeddings_path = Self::embeddings_path(&self.path);

        if !data_path.exists() {
            return Ok(Vec::new());
        }

        // Load entries
        let mut data_file = fs::File::open(&data_path).await?;
        let mut data = Vec::new();
        data_file.read_to_end(&mut data).await?;

        let entries: Vec<KnowledgeEntry> =
            bincode::deserialize(&data).map_err(|e| Error::storage(e.to_string()))?;

        // Load embeddings
        let embeddings: Vec<Vec<f32>> = if embeddings_path.exists() {
            let mut emb_file = fs::File::open(&embeddings_path).await?;
            let mut emb_data = Vec::new();
            emb_file.read_to_end(&mut emb_data).await?;
            bincode::deserialize(&emb_data).map_err(|e| Error::storage(e.to_string()))?
        } else {
            vec![Vec::new(); entries.len()]
        };

        Ok(entries.into_iter().zip(embeddings).collect())
    }

    /// Save a single entry with its embedding.
    pub async fn save_entry(&self, entry: &KnowledgeEntry, embedding: &[f32]) -> Result<()> {
        // Update index
        {
            let mut index = self.index.write().await;
            index.entries.insert(
                entry.id,
                EntryMetadata {
                    id: entry.id,
                    title: entry.title.clone(),
                    file_offset: 0,
                },
            );
        }

        *self.dirty.write().await = true;

        self.flush_internal(Some((entry.clone(), embedding.to_vec())))
            .await
    }

    /// Save multiple entries in batch.
    pub async fn save_batch(&self, batch: &[(KnowledgeEntry, Vec<f32>)]) -> Result<()> {
        {
            let mut index = self.index.write().await;
            for (entry, _) in batch {
                index.entries.insert(
                    entry.id,
                    EntryMetadata {
                        id: entry.id,
                        title: entry.title.clone(),
                        file_offset: 0,
                    },
                );
            }
        }

        *self.dirty.write().await = true;
        self.flush().await
    }

    /// Delete an entry.
    pub async fn delete_entry(&self, id: Uuid) -> Result<()> {
        {
            let mut index = self.index.write().await;
            index.entries.remove(&id);
        }

        *self.dirty.write().await = true;
        self.flush().await
    }

    /// Flush pending writes to disk.
    pub async fn flush(&self) -> Result<()> {
        self.flush_internal(None).await
    }

    /// Internal flush with optional new entry.
    async fn flush_internal(&self, new_entry: Option<(KnowledgeEntry, Vec<f32>)>) -> Result<()> {
        // Load existing data
        let mut all_data = self.load_all().await.unwrap_or_default();

        // Add or update new entry
        if let Some((entry, embedding)) = new_entry {
            if let Some(pos) = all_data.iter().position(|(e, _)| e.id == entry.id) {
                all_data[pos] = (entry, embedding);
            } else {
                all_data.push((entry, embedding));
            }
        }

        // Filter by current index - clone the index to avoid holding lock across await
        let index_snapshot = {
            let index = self.index.read().await;
            index.clone()
        };
        all_data.retain(|(e, _)| index_snapshot.entries.contains_key(&e.id));

        // Separate entries and embeddings
        let entries: Vec<_> = all_data.iter().map(|(e, _)| e.clone()).collect();
        let embeddings: Vec<_> = all_data.iter().map(|(_, emb)| emb.clone()).collect();

        // Write data file
        let data_path = Self::data_path(&self.path);
        let data = bincode::serialize(&entries).map_err(|e| Error::storage(e.to_string()))?;
        let mut file = fs::File::create(&data_path).await?;
        file.write_all(&data).await?;
        file.sync_all().await?;

        // Write embeddings file
        let embeddings_path = Self::embeddings_path(&self.path);
        let emb_data =
            bincode::serialize(&embeddings).map_err(|e| Error::storage(e.to_string()))?;
        let mut emb_file = fs::File::create(&embeddings_path).await?;
        emb_file.write_all(&emb_data).await?;
        emb_file.sync_all().await?;

        // Write index - serialize before async write
        let index_path = Self::index_path(&self.path);
        let index_data = {
            let index = self.index.read().await;
            bincode::serialize(&*index).map_err(|e| Error::storage(e.to_string()))?
        };

        let mut index_file = fs::File::create(&index_path).await?;
        index_file.write_all(&index_data).await?;
        index_file.sync_all().await?;

        *self.dirty.write().await = false;
        Ok(())
    }

    /// Get storage statistics.
    pub async fn stats(&self) -> StorageStats {
        let index = self.index.read().await;
        StorageStats {
            entry_count: index.entries.len(),
            version: index.version,
        }
    }
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Number of knowledge entries currently stored.
    pub entry_count: usize,
    /// Storage format version.
    pub version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let storage = StorageBackend::open(&path).await.unwrap();
        assert_eq!(storage.stats().await.entry_count, 0);
    }

    #[tokio::test]
    async fn test_storage_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");

        let storage = StorageBackend::open(&path).await.unwrap();

        let entry = KnowledgeEntry::new("Test", "Content");
        let embedding = vec![0.1, 0.2, 0.3];

        storage.save_entry(&entry, &embedding).await.unwrap();

        // Reload
        let storage2 = StorageBackend::open(&path).await.unwrap();
        let loaded = storage2.load_all().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0.title, "Test");
    }
}
