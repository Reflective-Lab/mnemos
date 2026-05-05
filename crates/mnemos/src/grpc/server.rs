//! gRPC server implementation.

use super::knowledge_service_server::KnowledgeService;
use super::*;
use crate::core::{KnowledgeBase, KnowledgeEntry as CoreEntry, SearchOptions};

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

/// gRPC service implementation for the knowledge base.
pub struct KnowledgeServiceImpl {
    kb: Arc<RwLock<KnowledgeBase>>,
    start_time: Instant,
}

impl KnowledgeServiceImpl {
    /// Create a new service instance.
    pub fn new(kb: KnowledgeBase) -> Self {
        Self {
            kb: Arc::new(RwLock::new(kb)),
            start_time: Instant::now(),
        }
    }

    /// Create from shared knowledge base.
    pub fn from_shared(kb: Arc<RwLock<KnowledgeBase>>) -> Self {
        Self {
            kb,
            start_time: Instant::now(),
        }
    }

    /// Convert core entry to proto entry.
    fn to_proto_entry(entry: &CoreEntry) -> KnowledgeEntry {
        KnowledgeEntry {
            id: entry.id.to_string(),
            title: entry.title.clone(),
            content: entry.content.clone(),
            category: entry.category.clone(),
            tags: entry.tags.clone(),
            source: entry.source.clone(),
            metadata: entry
                .metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            created_at: entry.created_at.to_rfc3339(),
            updated_at: entry.updated_at.to_rfc3339(),
            access_count: entry.access_count,
            learned_relevance: entry.learned_relevance,
            related_entries: entry
                .related_entries
                .iter()
                .map(|id| id.to_string())
                .collect(),
        }
    }

    /// Convert add request to core entry.
    fn from_add_request(req: &AddEntryRequest) -> CoreEntry {
        let mut entry = CoreEntry::new(&req.title, &req.content);

        if let Some(cat) = &req.category {
            entry = entry.with_category(cat);
        }

        if !req.tags.is_empty() {
            entry = entry.with_tags(req.tags.clone());
        }

        if let Some(src) = &req.source {
            entry = entry.with_source(src);
        }

        for (k, v) in &req.metadata {
            entry = entry.with_metadata(k, v);
        }

        entry
    }
}

#[tonic::async_trait]
impl KnowledgeService for KnowledgeServiceImpl {
    async fn add_entry(
        &self,
        request: Request<AddEntryRequest>,
    ) -> std::result::Result<Response<AddEntryResponse>, Status> {
        let req = request.into_inner();
        let entry = Self::from_add_request(&req);

        let kb = self.kb.read().await;
        match kb.add_entry(entry).await {
            Ok(id) => Ok(Response::new(AddEntryResponse {
                id: id.to_string(),
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(AddEntryResponse {
                id: String::new(),
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn add_entries(
        &self,
        request: Request<AddEntriesRequest>,
    ) -> std::result::Result<Response<AddEntriesResponse>, Status> {
        let req = request.into_inner();
        let entries: Vec<CoreEntry> = req.entries.iter().map(Self::from_add_request).collect();

        let kb = self.kb.read().await;
        match kb.add_entries(entries).await {
            Ok(ids) => Ok(Response::new(AddEntriesResponse {
                ids: ids.iter().map(|id| id.to_string()).collect(),
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(AddEntriesResponse {
                ids: Vec::new(),
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn get_entry(
        &self,
        request: Request<GetEntryRequest>,
    ) -> std::result::Result<Response<GetEntryResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;
        match kb.get(id) {
            Some(entry) => Ok(Response::new(GetEntryResponse {
                entry: Some(Self::to_proto_entry(&entry)),
                found: true,
            })),
            None => Ok(Response::new(GetEntryResponse {
                entry: None,
                found: false,
            })),
        }
    }

    async fn update_entry(
        &self,
        request: Request<UpdateEntryRequest>,
    ) -> std::result::Result<Response<UpdateEntryResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;

        // Get existing entry
        let mut entry = match kb.get(id) {
            Some(e) => e,
            None => {
                return Ok(Response::new(UpdateEntryResponse {
                    success: false,
                    error: Some("Entry not found".to_string()),
                }));
            }
        };

        // Update fields
        if let Some(title) = req.title {
            entry.title = title;
        }
        if let Some(content) = req.content {
            entry.content = content;
        }
        if let Some(category) = req.category {
            entry.category = Some(category);
        }
        if !req.tags.is_empty() {
            entry.tags = req.tags;
        }
        if let Some(source) = req.source {
            entry.source = Some(source);
        }
        for (k, v) in req.metadata {
            entry.metadata.insert(k, v);
        }

        match kb.update_entry(entry).await {
            Ok(_) => Ok(Response::new(UpdateEntryResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(UpdateEntryResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn delete_entry(
        &self,
        request: Request<DeleteEntryRequest>,
    ) -> std::result::Result<Response<DeleteEntryResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;
        match kb.delete_entry(id).await {
            Ok(_) => Ok(Response::new(DeleteEntryResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(DeleteEntryResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> std::result::Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let start = Instant::now();

        let options = SearchOptions {
            limit: req.limit as usize,
            min_similarity: req.min_similarity,
            category: req.category,
            tags: req.tags,
            use_learning: req.use_learning,
            include_related: req.include_related,
            diversity: req.diversity,
            hybrid: req.hybrid,
            keyword_weight: req.keyword_weight,
        };

        let kb = self.kb.read().await;
        match kb.search(&req.query, options).await {
            Ok(results) => {
                let elapsed = start.elapsed().as_secs_f32() * 1000.0;
                let proto_results: Vec<SearchResult> = results
                    .iter()
                    .map(|r| SearchResult {
                        entry: Some(Self::to_proto_entry(&r.entry)),
                        similarity: r.similarity,
                        relevance_boost: r.relevance_boost,
                        score: r.score,
                        distance: r.distance,
                    })
                    .collect();

                Ok(Response::new(SearchResponse {
                    results: proto_results.clone(),
                    total_results: proto_results.len() as u32,
                    search_time_ms: elapsed,
                }))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    type SearchStreamStream = ReceiverStream<std::result::Result<SearchResult, Status>>;

    async fn search_stream(
        &self,
        request: Request<SearchRequest>,
    ) -> std::result::Result<Response<Self::SearchStreamStream>, Status> {
        let req = request.into_inner();
        let kb = self.kb.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            let options = SearchOptions {
                limit: req.limit as usize,
                min_similarity: req.min_similarity,
                category: req.category,
                tags: req.tags,
                use_learning: req.use_learning,
                include_related: req.include_related,
                diversity: req.diversity,
                hybrid: req.hybrid,
                keyword_weight: req.keyword_weight,
            };

            let kb = kb.read().await;
            if let Ok(results) = kb.search(&req.query, options).await {
                for result in results {
                    let proto_result = SearchResult {
                        entry: Some(KnowledgeServiceImpl::to_proto_entry(&result.entry)),
                        similarity: result.similarity,
                        relevance_boost: result.relevance_boost,
                        score: result.score,
                        distance: result.distance,
                    };

                    if tx.send(Ok(proto_result)).await.is_err() {
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn record_feedback(
        &self,
        request: Request<FeedbackRequest>,
    ) -> std::result::Result<Response<FeedbackResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::parse_str(&req.entry_id)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;
        match kb.record_feedback(id, req.positive).await {
            Ok(_) => Ok(Response::new(FeedbackResponse { success: true })),
            Err(_) => Ok(Response::new(FeedbackResponse { success: false })),
        }
    }

    async fn get_related(
        &self,
        request: Request<GetRelatedRequest>,
    ) -> std::result::Result<Response<GetRelatedResponse>, Status> {
        let req = request.into_inner();
        let id = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;
        let related = kb.get_related(id, req.limit as usize);

        Ok(Response::new(GetRelatedResponse {
            entries: related.iter().map(Self::to_proto_entry).collect(),
        }))
    }

    async fn link_entries(
        &self,
        request: Request<LinkEntriesRequest>,
    ) -> std::result::Result<Response<LinkEntriesResponse>, Status> {
        let req = request.into_inner();
        let id1 = Uuid::parse_str(&req.id1)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;
        let id2 = Uuid::parse_str(&req.id2)
            .map_err(|e| Status::invalid_argument(format!("Invalid UUID: {}", e)))?;

        let kb = self.kb.read().await;
        match kb.link_entries(id1, id2).await {
            Ok(_) => Ok(Response::new(LinkEntriesResponse {
                success: true,
                error: None,
            })),
            Err(e) => Ok(Response::new(LinkEntriesResponse {
                success: false,
                error: Some(e.to_string()),
            })),
        }
    }

    async fn get_stats(
        &self,
        _request: Request<GetStatsRequest>,
    ) -> std::result::Result<Response<GetStatsResponse>, Status> {
        let kb = self.kb.read().await;
        let stats = kb.stats();

        Ok(Response::new(GetStatsResponse {
            total_entries: stats.total_entries as u64,
            unique_categories: stats.unique_categories as u64,
            unique_tags: stats.unique_tags as u64,
            total_access_count: stats.total_access_count,
            dimensions: stats.dimensions as u32,
            learning_enabled: stats.learning_enabled,
            learning_stats: None, // TODO: Add learning stats
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            healthy: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }))
    }
}
