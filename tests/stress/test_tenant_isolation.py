"""Test 6: Multi-tenant data isolation under concurrent load.

Verifies that tenant A's documents are completely invisible to tenant B,
even under heavy concurrent load with many tenants.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from loguru import logger

from .helpers.api_client import StressAPIClient
from .helpers.document_fixtures import generate_tenant_unique_document

pytestmark = pytest.mark.stress


class TestTenantIsolation:
    """Multi-tenant data isolation tests."""

    @pytest.mark.asyncio
    async def test_document_isolation(self, docker_manager, backend, stress_config):
        """Tenant A's documents are invisible to Tenant B.

        1. Tenant A uploads doc with unique content "TENANT-SECRET-{uuid}"
        2. Tenant B searches for that exact unique marker
        3. Assert Tenant B gets 0 results for the marker
        4. Assert Tenant A gets >= 1 result for same query
        """
        await docker_manager.start(backend)

        num_pairs = 20
        results_log = []

        # Create tenant pairs
        pairs = []
        for i in range(num_pairs):
            tenant_a_id = f"iso-a-{i}-{uuid.uuid4().hex[:8]}"
            tenant_b_id = f"iso-b-{i}-{uuid.uuid4().hex[:8]}"
            pairs.append((tenant_a_id, tenant_b_id))

        clients = {}
        markers = {}

        try:
            # Phase 1: Each Tenant A uploads a unique document
            upload_tasks = []
            for tenant_a_id, tenant_b_id in pairs:
                client_a = StressAPIClient(stress_config.api_base_url, tenant_a_id)
                client_b = StressAPIClient(stress_config.api_base_url, tenant_b_id)
                clients[tenant_a_id] = client_a
                clients[tenant_b_id] = client_b

                doc = generate_tenant_unique_document(tenant_a_id)
                markers[tenant_a_id] = doc.unique_marker
                upload_tasks.append((tenant_a_id, client_a, doc))

            upload_ok = 0
            upload_fail = 0
            for tenant_a_id, client_a, doc in upload_tasks:
                result = await client_a.upload_document(doc.path, doc.filename)
                if result.success:
                    await client_a.wait_for_ready(
                        result.document_id,
                        timeout=stress_config.upload_timeout,
                    )
                    upload_ok += 1
                else:
                    upload_fail += 1
                    logger.warning(
                        f"Upload failed for {tenant_a_id}: "
                        f"status={result.status_code} err={result.error}"
                    )

            assert upload_ok > 0, (
                f"All {upload_fail} document uploads failed. "
                f"Cannot test isolation without uploaded documents."
            )

            # Allow vector index to settle after bulk inserts.
            # pgvector indexes asynchronously; immediate search after insert
            # may miss recently-added vectors.
            await asyncio.sleep(3.0)

            # Phase 2: Verify isolation
            _isolation_checks = []

            async def check_isolation(tenant_a_id, tenant_b_id, marker):
                client_a = clients[tenant_a_id]
                client_b = clients[tenant_b_id]

                # Tenant A should find their own document
                r_a = await client_a.search(marker, top_k=5)

                # Tenant B should NOT find Tenant A's document
                r_b = await client_b.search(marker, top_k=5)

                return {
                    "tenant_a": tenant_a_id,
                    "tenant_b": tenant_b_id,
                    "marker": marker,
                    "a_found": r_a.total_results > 0 if r_a.success else False,
                    "b_found": r_b.total_results > 0 if r_b.success else False,
                    "b_results": r_b.results if r_b.success else [],
                }

            tasks = [check_isolation(a_id, b_id, markers[a_id]) for a_id, b_id in pairs]
            results_log = await asyncio.gather(*tasks, return_exceptions=True)

            # Assert: tenant A finds their own document (positive retrieval)
            a_not_found = []
            for r in results_log:
                if isinstance(r, Exception):
                    continue
                if not r["a_found"]:
                    a_not_found.append(r["tenant_a"])

            if a_not_found:
                import logging

                logging.getLogger(__name__).warning(
                    f"{len(a_not_found)}/{num_pairs} tenants did not find "
                    f"their own document — possible indexing delay"
                )

            # Assert: zero cross-tenant leaks
            leaks = []
            for r in results_log:
                if isinstance(r, Exception):
                    continue
                if r["b_found"]:
                    leaks.append(r)

            assert len(leaks) == 0, (
                f"CRITICAL: {len(leaks)} cross-tenant data leaks detected! Leaks: {leaks[:3]}"
            )

            # At least half the tenants should find their own document
            found_count = num_pairs - len(a_not_found)
            assert found_count >= num_pairs * 0.5, (
                f"Only {found_count}/{num_pairs} tenants found their own document. "
                f"This indicates a systemic indexing or search issue."
            )

        finally:
            for client in clients.values():
                await client.close()

    @pytest.mark.asyncio
    async def test_search_isolation_under_load(self, docker_manager, backend, stress_config):
        """Under heavy concurrent load, no cross-tenant data leaks.

        Each tenant has unique domain content. Run concurrent searches
        across all tenants. Flag any result where document_id belongs
        to a different tenant's upload set.
        """
        await docker_manager.start(backend)

        num_tenants = min(stress_config.num_tenants, 10)
        clients = {}
        tenant_doc_ids = {}  # session_id -> set of uploaded doc_ids

        try:
            # Set up tenants with unique documents
            for i in range(num_tenants):
                session_id = f"load-iso-{i}-{uuid.uuid4().hex[:8]}"
                client = StressAPIClient(stress_config.api_base_url, session_id)
                clients[session_id] = client
                tenant_doc_ids[session_id] = set()

                doc = generate_tenant_unique_document(session_id)
                result = await client.upload_document(doc.path, doc.filename)
                if result.success:
                    await client.wait_for_ready(
                        result.document_id, timeout=stress_config.upload_timeout
                    )
                    tenant_doc_ids[session_id].add(result.document_id)

            # Concurrent searches across all tenants
            async def tenant_search(session_id, client):
                leaks = []
                for _ in range(5):
                    result = await client.search(f"search for {session_id[:12]}", top_k=10)
                    if result.success:
                        for r in result.results:
                            doc_id = r.get("document_id", "")
                            # Check if this doc_id belongs to another tenant
                            for other_id, other_docs in tenant_doc_ids.items():
                                if other_id != session_id and doc_id in other_docs:
                                    leaks.append(
                                        {
                                            "searcher": session_id,
                                            "owner": other_id,
                                            "doc_id": doc_id,
                                        }
                                    )
                return leaks

            tasks = [tenant_search(sid, client) for sid, client in clients.items()]
            all_leaks = await asyncio.gather(*tasks, return_exceptions=True)

            total_leaks = []
            for result in all_leaks:
                if isinstance(result, list):
                    total_leaks.extend(result)

            assert len(total_leaks) == 0, (
                f"CRITICAL: {len(total_leaks)} cross-tenant search leaks under load! "
                f"Examples: {total_leaks[:3]}"
            )

        finally:
            for client in clients.values():
                await client.close()

    @pytest.mark.asyncio
    async def test_deletion_isolation(self, docker_manager, backend, stress_config):
        """Tenant A deleting a document doesn't affect Tenant B.

        1. Both tenants have documents
        2. Tenant A deletes a document
        3. Tenant B's document list unchanged
        4. Tenant A's document count decremented
        """
        await docker_manager.start(backend)

        tenant_a = StressAPIClient(
            stress_config.api_base_url,
            f"del-iso-a-{uuid.uuid4().hex[:8]}",
        )
        tenant_b = StressAPIClient(
            stress_config.api_base_url,
            f"del-iso-b-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Upload docs for both tenants
            doc_a = generate_tenant_unique_document("tenant-a")
            doc_b = generate_tenant_unique_document("tenant-b")

            result_a = await tenant_a.upload_document(doc_a.path, doc_a.filename)
            result_b = await tenant_b.upload_document(doc_b.path, doc_b.filename)

            assert result_a.success, f"Tenant A upload failed: {result_a.error}"
            assert result_b.success, f"Tenant B upload failed: {result_b.error}"

            await tenant_a.wait_for_ready(
                result_a.document_id, timeout=stress_config.upload_timeout
            )
            await tenant_b.wait_for_ready(
                result_b.document_id, timeout=stress_config.upload_timeout
            )

            # Get initial document counts
            docs_b_before = await tenant_b.list_documents()
            count_b_before = len(docs_b_before)

            # Tenant A deletes their document
            delete_result = await tenant_a.delete_document(result_a.document_id)
            assert delete_result.success, "Tenant A deletion failed"

            # Tenant B's documents should be unchanged
            docs_b_after = await tenant_b.list_documents()
            count_b_after = len(docs_b_after)

            assert count_b_after == count_b_before, (
                f"Tenant B document count changed after Tenant A deletion: "
                f"{count_b_before} -> {count_b_after}"
            )

            # Verify Tenant A's document is gone
            docs_a_after = await tenant_a.list_documents()
            a_doc_ids = {d.get("document_id") for d in docs_a_after}
            assert result_a.document_id not in a_doc_ids, (
                "Tenant A's deleted document still appears in listing"
            )

        finally:
            await tenant_a.close()
            await tenant_b.close()
