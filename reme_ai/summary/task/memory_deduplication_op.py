"""Memory deduplication operation for task memory management.

This module provides operations to remove duplicate or highly similar task
memories by comparing embeddings and calculating similarity scores.
"""

from typing import List

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from loguru import logger

from reme_ai.schema.memory import BaseMemory


@C.register_op()
class MemoryDeduplicationOp(BaseAsyncOp):
    """Remove duplicate task memories using embedding similarity.

    This operation identifies and removes duplicate or highly similar task
    memories by comparing their embeddings against both existing memories
    in the vector store and other memories in the current batch.
    """

    file_path: str = __file__

    async def async_execute(self):
        """Remove duplicate task memories"""
        # Get task memories to deduplicate
        task_memories: List[BaseMemory] = self.context.response.metadata.get("memory_list", [])

        if not task_memories:
            logger.info("No task memories found for deduplication")
            return

        logger.info(f"Starting deduplication for {len(task_memories)} task memories")

        # Perform deduplication
        deduplicated_task_memories = await self._deduplicate_task_memories(task_memories)

        logger.info(
            f"Deduplication complete: {len(deduplicated_task_memories)} deduplicated "
            f"task memories out of {len(task_memories)}",
        )

        # Update context
        self.context.response.metadata["memory_list"] = deduplicated_task_memories
        self.context.response.metadata["dedup_decisions"] = getattr(self, "_last_dedup_decisions", [])

    async def _deduplicate_task_memories(self, task_memories: List[BaseMemory]) -> List[BaseMemory]:
        """Remove duplicate task memories"""
        if not task_memories:
            return task_memories

        similarity_threshold = self.op_params.get("similarity_threshold", 0.5)
        workspace_id = self.context.get("workspace_id")

        unique_task_memories = []
        unique_embeddings = []
        self._last_dedup_decisions = []

        # Get existing task memory embeddings
        existing_embeddings = await self._get_existing_task_memory_embeddings(workspace_id)

        for task_memory in task_memories:
            # Generate embedding for current task memory
            current_embedding = self._get_task_memory_embedding(task_memory)

            if current_embedding is None:
                logger.warning(f"Failed to generate embedding for task memory: {str(task_memory.when_to_use)[:50]}...")
                continue

            # Check similarity with existing task memories
            existing_match = self._find_similar_embedding(current_embedding, existing_embeddings, similarity_threshold)
            if existing_match is not None:
                logger.debug(f"Skipping similar task memory: {str(task_memory.when_to_use)[:50]}...")
                self._last_dedup_decisions.append(
                    {
                        "memory_id": task_memory.memory_id,
                        "decision": "duplicate_existing",
                        "matched_memory_id": existing_match[0],
                        "similarity": existing_match[1],
                    },
                )
                continue

            # Check similarity with current batch task memories
            batch_match = self._find_similar_embedding(current_embedding, unique_embeddings, similarity_threshold)
            if batch_match is not None:
                logger.debug(f"Skipping duplicate in current batch: {str(task_memory.when_to_use)[:50]}...")
                self._last_dedup_decisions.append(
                    {
                        "memory_id": task_memory.memory_id,
                        "decision": "duplicate_batch",
                        "matched_memory_id": batch_match[0],
                        "similarity": batch_match[1],
                    },
                )
                continue

            # Add to unique task memories list
            unique_task_memories.append(task_memory)
            unique_embeddings.append((task_memory.memory_id, current_embedding))
            self._last_dedup_decisions.append(
                {
                    "memory_id": task_memory.memory_id,
                    "decision": "keep",
                    "matched_memory_id": None,
                    "similarity": None,
                },
            )
            logger.debug(f"Added unique task memory: {str(task_memory.when_to_use)[:50]}...")

        return unique_task_memories

    async def _get_existing_task_memory_embeddings(self, workspace_id: str) -> List[tuple[str, List[float]]]:
        """Get embeddings of existing task memories"""
        try:
            if not hasattr(self, "vector_store") or not self.vector_store or not workspace_id:
                return []

            existing_nodes = await self.vector_store.async_list_workspace_nodes(workspace_id=workspace_id)

            # Extract embeddings
            existing_embeddings = []
            for node in existing_nodes:
                if node.metadata.get("memory_type") != "task":
                    continue
                embedding = getattr(node, "embedding", None)
                if embedding is None or len(embedding) == 0:
                    stored_content = node.metadata.get("content", "")
                    text_for_embedding = f"{node.content} {stored_content}"
                    generated = self.vector_store.embedding_model.get_embeddings([text_for_embedding])
                    embedding = generated[0] if generated else None
                if embedding is not None and len(embedding) > 0:
                    existing_embeddings.append((node.unique_id, embedding))

            logger.debug(
                f"Retrieved {len(existing_embeddings)} existing task memory embeddings from workspace {workspace_id}",
            )
            return existing_embeddings

        except Exception as e:
            # Failing open here can silently insert duplicates. A2 treats the
            # workspace-wide duplicate check as part of the commit boundary.
            logger.error(f"Failed to retrieve existing task memory embeddings: {e}")
            raise RuntimeError("workspace-wide memory deduplication failed") from e

    def _get_task_memory_embedding(self, task_memory: BaseMemory) -> List[float] | None:
        """Generate embedding for task memory"""
        try:

            # Combine task memory description and content for embedding
            text_for_embedding = f"{task_memory.when_to_use} {task_memory.content}"
            embeddings = self.vector_store.embedding_model.get_embeddings([text_for_embedding])

            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.warning("Empty embedding generated for task memory")
                return None

        except Exception as e:
            logger.error(f"Error generating embedding for task memory: {e}")
            return None

    def _find_similar_embedding(
        self,
        current_embedding: List[float],
        existing_embeddings: List[tuple[str, List[float]]],
        threshold: float,
    ) -> tuple[str, float] | None:
        """Return the first matching memory id and similarity."""
        for memory_id, existing_embedding in existing_embeddings:
            similarity = self._calculate_cosine_similarity(current_embedding, existing_embedding)
            if similarity > threshold:
                logger.debug(f"Found similar existing task memory with similarity: {similarity:.3f}")
                return memory_id, float(similarity)
        return None

    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            import numpy as np

            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
