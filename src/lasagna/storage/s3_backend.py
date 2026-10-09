"""S3-compatible storage backend.

Works with AWS S3, MinIO, SeaweedFS, Ceph, Garage, RustFS, Cloudflare R2,
Backblaze B2, Wasabi, DigitalOcean Spaces, and 15+ other S3-compatible
services via configurable ``endpoint_url``.

Requires: ``pip install boto3>=1.28`` (or ``pip install lasagna-rag[storage-s3]``)

Usage::

    from lasagna.storage import get_storage_backend

    # AWS S3
    storage = get_storage_backend("s3", bucket="my-docs")

    # MinIO / R2 / any S3-compatible
    storage = get_storage_backend(
        "s3",
        bucket="my-docs",
        endpoint_url="http://minio:9000",
    )
"""

import io
from typing import BinaryIO, Dict, Optional

from loguru import logger

from .base import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    StorageError,
    StorageNotFoundError,
    StorageConnectionError,
)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3StorageBackend(StorageBackend):
    """S3-compatible file storage backend.

    Supports AWS S3, MinIO, R2, Backblaze B2, Wasabi, DigitalOcean Spaces,
    and any service that speaks the S3 protocol.

    URI format: ``s3://<bucket>/<prefix>/<document_id>/<filename>``
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "rag-documents",
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name (must already exist)
            prefix: Key prefix for all stored files
            endpoint_url: S3-compatible endpoint (MinIO, R2, etc.)
                         None = standard AWS S3
            region_name: AWS region (default: us-east-1)
            aws_access_key_id: Optional explicit credentials
                              (otherwise uses default credential chain)
            aws_secret_access_key: Optional explicit credentials
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "S3 storage backend requires boto3. "
                "Install with: pip install boto3 (or pip install lasagna-rag[storage-s3])"
            )

        self._bucket = bucket
        self._prefix = prefix.strip("/")

        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        client_kwargs = {"region_name": region_name}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        try:
            session = boto3.Session(**session_kwargs)
            self._client = session.client("s3", **client_kwargs)
        except NoCredentialsError as e:
            raise StorageConnectionError(f"S3 credentials not found: {e}") from e

        self._endpoint_url = endpoint_url
        logger.info(
            f"S3 storage initialized: bucket={bucket}, prefix={prefix}, "
            f"endpoint={endpoint_url or 'AWS S3'}"
        )

    @property
    def backend_name(self) -> str:
        return "s3"

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_presigned_urls=True,
            supports_streaming=True,
            supports_multipart_upload=True,
            max_file_size=5 * 1024**4,  # 5 TB (S3 limit)
        )

    def _key(self, document_id: str, filename: str) -> str:
        """Build the S3 object key."""
        return f"{self._prefix}/{document_id}/{filename}"

    def _key_from_uri(self, uri: str) -> str:
        """Extract S3 key from a ``s3://bucket/key`` URI."""
        if uri.startswith("s3://"):
            # s3://bucket/prefix/doc_id/file.pdf -> prefix/doc_id/file.pdf
            parts = uri[5:]  # strip "s3://"
            # Remove bucket name
            _, _, key = parts.partition("/")
            return key
        return uri

    def _to_uri(self, key: str) -> str:
        """Build URI from S3 key."""
        return f"s3://{self._bucket}/{key}"

    # =========================================================================
    # Core Operations
    # =========================================================================

    def store(
        self,
        source: BinaryIO,
        document_id: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageFileInfo:
        """Upload file to S3."""
        self._validate_store_inputs(document_id, filename)
        key = self._key(document_id, filename)
        extra_args = {}

        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        try:
            self._client.upload_fileobj(source, self._bucket, key, ExtraArgs=extra_args or None)
        except ClientError as e:
            raise StorageError(f"S3 upload failed: {e}") from e

        # Get size
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            size = head["ContentLength"]
            etag = head.get("ETag", "").strip('"')
        except ClientError:
            size = 0
            etag = None

        uri = self._to_uri(key)
        logger.info(f"File stored in S3: {uri}")
        return StorageFileInfo(
            uri=uri,
            size=size,
            content_type=content_type,
            etag=etag,
            metadata=metadata or {},
        )

    def retrieve(self, uri: str) -> BinaryIO:
        """Download file from S3 into an in-memory buffer."""
        key = self._key_from_uri(uri)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            buf = io.BytesIO(response["Body"].read())
            buf.seek(0)
            return buf
        except ClientError as e:
            http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            error_code = e.response.get("Error", {}).get("Code", "")
            if http_status == 404 or error_code in ("NoSuchKey", "404", "Not Found"):
                raise StorageNotFoundError(f"File not found in S3: {uri}") from e
            raise StorageError(f"S3 retrieve failed: {e}") from e

    def delete(self, uri: str) -> bool:
        """Delete a file (and all files under the same document prefix) from S3."""
        key = self._key_from_uri(uri)
        # Delete the document directory (all objects under doc_id prefix)
        doc_prefix = "/".join(key.split("/")[:-1]) + "/"

        try:
            # Paginate through all objects under the document prefix
            total_deleted = 0
            continuation_token = None

            while True:
                list_kwargs = {"Bucket": self._bucket, "Prefix": doc_prefix}
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self._client.list_objects_v2(**list_kwargs)
                objects = response.get("Contents", [])

                if not objects and total_deleted == 0:
                    return False

                if objects:
                    delete_objects = [{"Key": obj["Key"]} for obj in objects]
                    self._client.delete_objects(
                        Bucket=self._bucket, Delete={"Objects": delete_objects}
                    )
                    total_deleted += len(delete_objects)

                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

            logger.info(f"Deleted {total_deleted} objects from S3: {doc_prefix}")
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    def exists(self, uri: str) -> bool:
        """Check if a file exists in S3."""
        if not uri:
            return False
        key = self._key_from_uri(uri)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_size(self, uri: str) -> int:
        """Get file size from S3."""
        key = self._key_from_uri(uri)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            return head["ContentLength"]
        except ClientError as e:
            raise StorageNotFoundError(f"File not found in S3: {uri}") from e

    # =========================================================================
    # S3-Specific Methods
    # =========================================================================

    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a pre-signed download URL."""
        key = self._key_from_uri(uri)
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError:
            return None
