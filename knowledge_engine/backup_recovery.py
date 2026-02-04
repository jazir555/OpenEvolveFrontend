"""
Backup and Disaster Recovery System

Provides comprehensive backup and recovery capabilities:
- Automated scheduled backups
- Incremental and full backups
- Point-in-time recovery
- Cross-region replication
- Backup verification
- Disaster recovery procedures
- Multi-cloud storage support (S3, GCS, Azure, Local)
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    source_path: str = ""
    destination_path: str = ""
    size_bytes: int = 0
    checksum: str = ""
    parent_backup_id: Optional[str] = None  # For incremental
    included_items: List[str] = field(default_factory=list)
    excluded_items: List[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    error_message: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "parent_backup_id": self.parent_backup_id,
            "compression_ratio": self.compression_ratio,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None
        }


@dataclass
class RecoveryPoint:
    """A point-in-time recovery point."""
    recovery_point_id: str
    backup_id: str
    timestamp: datetime
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupStorage:
    """
    Abstract base for backup storage backends.

    All storage backends must implement these methods to provide
    backup storage functionality with proper error handling and logging.
    """

    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: BackupMetadata
    ) -> str:
        """
        Store backup data. Returns storage identifier.

        Args:
            backup_id: Unique identifier for the backup
            data: Compressed backup data
            metadata: Backup metadata object

        Returns:
            Storage identifier (URL or path)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Storage backend must implement store() method")

    async def retrieve(self, storage_id: str) -> bytes:
        """
        Retrieve backup data.

        Args:
            storage_id: Storage identifier returned by store()

        Returns:
            Backup data bytes

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Storage backend must implement retrieve() method")

    async def delete(self, storage_id: str) -> bool:
        """
        Delete backup data.

        Args:
            storage_id: Storage identifier returned by store()

        Returns:
            True if successful, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Storage backend must implement delete() method")

    async def list_backups(self) -> List[str]:
        """
        List available backups.

        Returns:
            List of storage identifiers

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Storage backend must implement list_backups() method")


class LocalBackupStorage(BackupStorage):
    """Local filesystem backup storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info({
            "msg": "Local backup storage initialized",
            "base_path": str(self.base_path)
        })

    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: BackupMetadata
    ) -> str:
        backup_dir = self.base_path / backup_id
        backup_dir.mkdir(exist_ok=True)

        # Store data
        data_file = backup_dir / "data.gz"
        with gzip.open(data_file, 'wb') as f:
            f.write(data)

        # Store metadata
        meta_file = backup_dir / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info({
            "msg": "Backup stored locally",
            "backup_id": backup_id,
            "path": str(backup_dir),
            "size_bytes": len(data)
        })

        return str(backup_dir)

    async def retrieve(self, storage_id: str) -> bytes:
        data_file = Path(storage_id) / "data.gz"
        with gzip.open(data_file, 'rb') as f:
            data = f.read()

        logger.info({
            "msg": "Backup retrieved from local storage",
            "storage_id": storage_id,
            "size_bytes": len(data)
        })

        return data

    async def delete(self, storage_id: str) -> bool:
        try:
            shutil.rmtree(storage_id)
            logger.info({
                "msg": "Backup deleted from local storage",
                "storage_id": storage_id
            })
            return True
        except Exception as e:
            logger.error({
                "msg": "Failed to delete local backup",
                "storage_id": storage_id,
                "error": str(e)
            })
            return False

    async def list_backups(self) -> List[str]:
        backups = [str(d) for d in self.base_path.iterdir() if d.is_dir()]
        logger.info({
            "msg": "Listed local backups",
            "count": len(backups)
        })
        return backups


class S3BackupStorage(BackupStorage):
    """AWS S3 backup storage implementation."""

    def __init__(
        self,
        bucket_name: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        prefix: str = "backups/",
        storage_class: str = "STANDARD"
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.storage_class = storage_class
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL")
        self._s3_client = None
        self._client_error = None
        self._init_client()

    def _init_client(self):
        """Initialize S3 client with graceful degradation."""
        try:
            import boto3
            from botocore.exceptions import ClientError

            session = boto3.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region
            )

            kwargs = {}
            if self.endpoint_url:
                kwargs['endpoint_url'] = self.endpoint_url

            self._s3_client = session.client('s3', **kwargs)
            self._client_error = ClientError

            # Verify bucket exists
            try:
                self._s3_client.head_bucket(Bucket=self.bucket_name)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == '404':
                    logger.warning({
                        "msg": "S3 bucket not found",
                        "bucket": self.bucket_name
                    })

            logger.info({
                "msg": "S3 storage initialized",
                "bucket": self.bucket_name,
                "region": self.region
            })

        except ImportError as e:
            logger.error({
                "msg": "boto3 not available",
                "error": "boto3 is required for S3 storage. Install with: pip install boto3"
            })
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            ) from e
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize S3 client",
                "error": str(e)
            })
            raise

    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: BackupMetadata
    ) -> str:
        """Store backup data in S3."""
        key = f"{self.prefix}{backup_id}/data.gz"
        metadata_key = f"{self.prefix}{backup_id}/metadata.json"

        try:
            # Store compressed data
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                StorageClass=self.storage_class,
                Metadata={
                    'backup-id': backup_id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )

            # Store metadata
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata.to_dict()).encode(),
                ContentType='application/json'
            )

            storage_id = f"s3://{self.bucket_name}/{key}"
            logger.info({
                "msg": "Backup stored to S3",
                "backup_id": backup_id,
                "key": key,
                "size_bytes": len(data)
            })

            return storage_id

        except self._client_error as e:
            logger.error({
                "msg": "S3 store failed",
                "backup_id": backup_id,
                "error": str(e)
            })
            raise
        except Exception as e:
            logger.error({
                "msg": "Unexpected error storing to S3",
                "backup_id": backup_id,
                "error": str(e)
            })
            raise

    async def retrieve(self, storage_id: str) -> bytes:
        """Retrieve backup data from S3."""
        # Parse s3://bucket/key format
        if storage_id.startswith("s3://"):
            storage_id = storage_id[5:]

        if "/" in storage_id:
            bucket, key = storage_id.split("/", 1)
        else:
            bucket = self.bucket_name
            key = storage_id

        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()

            logger.info({
                "msg": "Backup retrieved from S3",
                "key": key,
                "size_bytes": len(data)
            })

            return data

        except self._client_error as e:
            logger.error({
                "msg": "S3 retrieve failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            raise
        except Exception as e:
            logger.error({
                "msg": "Unexpected error retrieving from S3",
                "storage_id": storage_id,
                "error": str(e)
            })
            raise

    async def delete(self, storage_id: str) -> bool:
        """Delete backup data from S3."""
        if storage_id.startswith("s3://"):
            storage_id = storage_id[5:]

        if "/" in storage_id:
            bucket, key = storage_id.split("/", 1)
        else:
            bucket = self.bucket_name
            key = storage_id

        try:
            self._s3_client.delete_object(Bucket=bucket, Key=key)

            # Also try to delete metadata
            metadata_key = key.replace("/data.gz", "/metadata.json")
            try:
                self._s3_client.delete_object(Bucket=bucket, Key=metadata_key)
            except:
                pass

            logger.info({
                "msg": "Backup deleted from S3",
                "key": key
            })
            return True

        except self._client_error as e:
            logger.error({
                "msg": "S3 delete failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            return False
        except Exception as e:
            logger.error({
                "msg": "Unexpected error deleting from S3",
                "storage_id": storage_id,
                "error": str(e)
            })
            return False

    async def list_backups(self) -> List[str]:
        """List available backups in S3."""
        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )

            backups = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('/data.gz'):
                    backup_id = key[len(self.prefix):].replace('/data.gz', '')
                    backups.append(f"s3://{self.bucket_name}/{key}")

            logger.info({
                "msg": "Listed S3 backups",
                "count": len(backups)
            })

            return backups

        except self._client_error as e:
            logger.error({
                "msg": "S3 list failed",
                "error": str(e)
            })
            return []
        except Exception as e:
            logger.error({
                "msg": "Unexpected error listing S3 backups",
                "error": str(e)
            })
            return []


class GCSBackupStorage(BackupStorage):
    """Google Cloud Storage backup storage implementation."""

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        credentials_json: Optional[str] = None,
        prefix: str = "backups/"
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.project_id = project_id or os.getenv("GCS_PROJECT_ID")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.credentials_json = credentials_json or os.getenv("GCS_CREDENTIALS_JSON")
        self._client = None
        self._bucket = None
        self._init_client()

    def _init_client(self):
        """Initialize GCS client with graceful degradation."""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account

            if self.credentials_path:
                # Load from service account file
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._client = storage.Client(
                    project=self.project_id,
                    credentials=creds
                )
            elif self.credentials_json:
                # Load from JSON string
                creds_info = json.loads(self.credentials_json)
                creds = service_account.Credentials.from_service_account_info(
                    creds_info
                )
                self._client = storage.Client(
                    project=self.project_id,
                    credentials=creds
                )
            else:
                # Use default credentials
                self._client = storage.Client(project=self.project_id)

            self._bucket = self._client.bucket(self.bucket_name)

            logger.info({
                "msg": "GCS storage initialized",
                "bucket": self.bucket_name,
                "project": self.project_id
            })

        except ImportError as e:
            logger.error({
                "msg": "google-cloud-storage not available",
                "error": "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            })
            raise ImportError(
                "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            ) from e
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize GCS client",
                "error": str(e)
            })
            raise

    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: BackupMetadata
    ) -> str:
        """Store backup data in GCS."""
        key = f"{self.prefix}{backup_id}/data.gz"
        metadata_key = f"{self.prefix}{backup_id}/metadata.json"

        try:
            # Store data
            blob = self._bucket.blob(key)
            blob.metadata = {
                'backup-id': backup_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            blob.upload_from_string(data)

            # Store metadata
            meta_blob = self._bucket.blob(metadata_key)
            meta_blob.upload_from_string(
                json.dumps(metadata.to_dict()),
                content_type='application/json'
            )

            storage_id = f"gs://{self.bucket_name}/{key}"
            logger.info({
                "msg": "Backup stored to GCS",
                "backup_id": backup_id,
                "key": key,
                "size_bytes": len(data)
            })

            return storage_id

        except Exception as e:
            logger.error({
                "msg": "GCS store failed",
                "backup_id": backup_id,
                "error": str(e)
            })
            raise

    async def retrieve(self, storage_id: str) -> bytes:
        """Retrieve backup data from GCS."""
        if storage_id.startswith("gs://"):
            storage_id = storage_id[5:]

        if "/" in storage_id:
            bucket_name, key = storage_id.split("/", 1)
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = storage_id

        try:
            blob = bucket.blob(key)
            data = blob.download_as_bytes()

            logger.info({
                "msg": "Backup retrieved from GCS",
                "key": key,
                "size_bytes": len(data)
            })

            return data

        except Exception as e:
            logger.error({
                "msg": "GCS retrieve failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            raise

    async def delete(self, storage_id: str) -> bool:
        """Delete backup data from GCS."""
        if storage_id.startswith("gs://"):
            storage_id = storage_id[5:]

        if "/" in storage_id:
            bucket_name, key = storage_id.split("/", 1)
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = storage_id

        try:
            blob = bucket.blob(key)
            blob.delete()

            # Also try to delete metadata
            metadata_key = key.replace("/data.gz", "/metadata.json")
            try:
                meta_blob = bucket.blob(metadata_key)
                meta_blob.delete()
            except:
                pass

            logger.info({
                "msg": "Backup deleted from GCS",
                "key": key
            })
            return True

        except Exception as e:
            logger.error({
                "msg": "GCS delete failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            return False

    async def list_backups(self) -> List[str]:
        """List available backups in GCS."""
        try:
            blobs = self._client.list_blobs(
                self.bucket_name,
                prefix=self.prefix
            )

            backups = []
            for blob in blobs:
                if blob.name.endswith('/data.gz'):
                    storage_id = f"gs://{self.bucket_name}/{blob.name}"
                    backups.append(storage_id)

            logger.info({
                "msg": "Listed GCS backups",
                "count": len(backups)
            })

            return backups

        except Exception as e:
            logger.error({
                "msg": "GCS list failed",
                "error": str(e)
            })
            return []


class AzureBackupStorage(BackupStorage):
    """Azure Blob Storage backup storage implementation."""

    def __init__(
        self,
        container_name: str,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        prefix: str = "backups/"
    ):
        self.container_name = container_name
        self.prefix = prefix
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_KEY")
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self._blob_service = None
        self._container_client = None
        self._init_client()

    def _init_client(self):
        """Initialize Azure Blob client with graceful degradation."""
        try:
            from azure.storage.blob import BlobServiceClient

            if self.connection_string:
                self._blob_service = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            elif self.account_key:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._blob_service = BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key
                )
            else:
                # Use default Azure credentials
                from azure.identity import DefaultAzureCredential
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._blob_service = BlobServiceClient(
                    account_url=account_url,
                    credential=DefaultAzureCredential()
                )

            self._container_client = self._blob_service.get_container_client(
                self.container_name
            )

            # Create container if it doesn't exist
            try:
                self._container_client.create_container()
            except:
                pass  # Container may already exist

            logger.info({
                "msg": "Azure storage initialized",
                "container": self.container_name,
                "account": self.account_name
            })

        except ImportError as e:
            logger.error({
                "msg": "azure-storage-blob not available",
                "error": "azure-storage-blob is required. Install with: pip install azure-storage-blob"
            })
            raise ImportError(
                "azure-storage-blob is required. Install with: pip install azure-storage-blob"
            ) from e
        except Exception as e:
            logger.error({
                "msg": "Failed to initialize Azure client",
                "error": str(e)
            })
            raise

    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: BackupMetadata
    ) -> str:
        """Store backup data in Azure Blob Storage."""
        key = f"{self.prefix}{backup_id}/data.gz"
        metadata_key = f"{self.prefix}{backup_id}/metadata.json"

        try:
            # Store data
            blob_client = self._container_client.get_blob_client(key)
            blob_client.upload_blob(data, overwrite=True)

            # Store metadata
            meta_blob = self._container_client.get_blob_client(metadata_key)
            meta_blob.upload_blob(
                json.dumps(metadata.to_dict()).encode(),
                overwrite=True
            )

            storage_id = f"azure://{self.container_name}/{key}"
            logger.info({
                "msg": "Backup stored to Azure",
                "backup_id": backup_id,
                "key": key,
                "size_bytes": len(data)
            })

            return storage_id

        except Exception as e:
            logger.error({
                "msg": "Azure store failed",
                "backup_id": backup_id,
                "error": str(e)
            })
            raise

    async def retrieve(self, storage_id: str) -> bytes:
        """Retrieve backup data from Azure."""
        if storage_id.startswith("azure://"):
            storage_id = storage_id[8:]

        if "/" in storage_id:
            container, key = storage_id.split("/", 1)
            container_client = self._blob_service.get_container_client(container)
        else:
            container_client = self._container_client
            key = storage_id

        try:
            blob_client = container_client.get_blob_client(key)
            data = blob_client.download_blob().readall()

            logger.info({
                "msg": "Backup retrieved from Azure",
                "key": key,
                "size_bytes": len(data)
            })

            return data

        except Exception as e:
            logger.error({
                "msg": "Azure retrieve failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            raise

    async def delete(self, storage_id: str) -> bool:
        """Delete backup data from Azure."""
        if storage_id.startswith("azure://"):
            storage_id = storage_id[8:]

        if "/" in storage_id:
            container, key = storage_id.split("/", 1)
            container_client = self._blob_service.get_container_client(container)
        else:
            container_client = self._container_client
            key = storage_id

        try:
            blob_client = container_client.get_blob_client(key)
            blob_client.delete_blob()

            # Also try to delete metadata
            metadata_key = key.replace("/data.gz", "/metadata.json")
            try:
                meta_blob = container_client.get_blob_client(metadata_key)
                meta_blob.delete_blob()
            except:
                pass

            logger.info({
                "msg": "Backup deleted from Azure",
                "key": key
            })
            return True

        except Exception as e:
            logger.error({
                "msg": "Azure delete failed",
                "storage_id": storage_id,
                "error": str(e)
            })
            return False

    async def list_backups(self) -> List[str]:
        """List available backups in Azure."""
        try:
            blobs = self._container_client.list_blobs(name_starts_with=self.prefix)

            backups = []
            for blob in blobs:
                if blob.name.endswith('/data.gz'):
                    storage_id = f"azure://{self.container_name}/{blob.name}"
                    backups.append(storage_id)

            logger.info({
                "msg": "Listed Azure backups",
                "count": len(backups)
            })

            return backups

        except Exception as e:
            logger.error({
                "msg": "Azure list failed",
                "error": str(e)
            })
            return []


def create_storage_backend(
    storage_type: str = "local",
    **kwargs
) -> BackupStorage:
    """
    Factory function to create backup storage backend.

    Supports multiple storage backends with graceful degradation:
    - local: Local filesystem storage (default, always available)
    - s3: AWS S3 or S3-compatible storage (requires boto3)
    - gcs: Google Cloud Storage (requires google-cloud-storage)
    - azure: Azure Blob Storage (requires azure-storage-blob)

    Args:
        storage_type: Type of storage backend ('local', 's3', 'gcs', 'azure')
        **kwargs: Configuration options for the specific backend

    Returns:
        BackupStorage instance

    Raises:
        ValueError: If storage_type is unsupported
        ImportError: If required dependencies are missing

    Examples:
        # Local storage
        storage = create_storage_backend('local', base_path='./backups')

        # S3 storage
        storage = create_storage_backend('s3', bucket_name='my-bucket')

        # GCS storage
        storage = create_storage_backend('gcs', bucket_name='my-bucket')

        # Azure storage
        storage = create_storage_backend('azure', container_name='my-container')
    """
    storage_type = storage_type.lower()

    correlation_id = str(uuid.uuid4())
    logger.info({
        "msg": "Creating storage backend",
        "storage_type": storage_type,
        "correlation_id": correlation_id,
        "kwargs": list(kwargs.keys())
    })

    try:
        if storage_type == 'local':
            base_path = kwargs.get('base_path', './backups')
            return LocalBackupStorage(base_path=base_path)

        elif storage_type == 's3':
            return S3BackupStorage(
                bucket_name=kwargs.get('bucket_name'),
                access_key_id=kwargs.get('access_key_id'),
                secret_access_key=kwargs.get('secret_access_key'),
                region=kwargs.get('region', 'us-east-1'),
                endpoint_url=kwargs.get('endpoint_url'),
                prefix=kwargs.get('prefix', 'backups/'),
                storage_class=kwargs.get('storage_class', 'STANDARD')
            )

        elif storage_type == 'gcs':
            return GCSBackupStorage(
                bucket_name=kwargs.get('bucket_name'),
                project_id=kwargs.get('project_id'),
                credentials_path=kwargs.get('credentials_path'),
                credentials_json=kwargs.get('credentials_json'),
                prefix=kwargs.get('prefix', 'backups/')
            )

        elif storage_type == 'azure':
            return AzureBackupStorage(
                container_name=kwargs.get('container_name'),
                account_name=kwargs.get('account_name'),
                account_key=kwargs.get('account_key'),
                connection_string=kwargs.get('connection_string'),
                prefix=kwargs.get('prefix', 'backups/')
            )

        else:
            raise ValueError(
                f"Unsupported storage type: {storage_type}. "
                f"Supported types: local, s3, gcs, azure"
            )

    except ImportError as e:
        logger.error({
            "msg": "Failed to create storage backend - missing dependencies",
            "storage_type": storage_type,
            "correlation_id": correlation_id,
            "error": str(e)
        })
        raise
    except Exception as e:
        logger.error({
            "msg": "Failed to create storage backend",
            "storage_type": storage_type,
            "correlation_id": correlation_id,
            "error": str(e)
        })
        raise


class BackupEngine:
    """
    Main backup engine.
    """
    
    def __init__(
        self,
        storage: BackupStorage,
        retention_days: int = 30,
        compression_level: int = 6
    ):
        self.storage = storage
        self.retention_days = retention_days
        self.compression_level = compression_level
        
        self._backups: Dict[str, BackupMetadata] = {}
        self._recovery_points: List[RecoveryPoint] = []
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        parent_backup_id: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> BackupMetadata:
        """
        Create a new backup.
        
        Args:
            source_path: Path to backup
            backup_type: Type of backup
            parent_backup_id: Parent backup for incremental
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
        """
        backup_id = str(uuid.uuid4())
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.RUNNING,
            started_at=datetime.utcnow(),
            source_path=source_path,
            included_items=include_patterns or [],
            excluded_items=exclude_patterns or []
        )
        
        if backup_type == BackupType.INCREMENTAL and parent_backup_id:
            metadata.parent_backup_id = parent_backup_id
        
        try:
            # Collect files to backup
            files_to_backup = self._collect_files(
                source_path,
                include_patterns,
                exclude_patterns
            )
            
            # For incremental, filter by modification time
            if backup_type == BackupType.INCREMENTAL and parent_backup_id:
                parent = self._backups.get(parent_backup_id)
                if parent:
                    files_to_backup = self._filter_incremental(
                        files_to_backup,
                        parent.started_at
                    )
            
            # Create backup archive
            backup_data = await self._create_archive(files_to_backup)
            
            # Calculate checksum
            metadata.checksum = hashlib.sha256(backup_data).hexdigest()
            metadata.size_bytes = len(backup_data)
            
            # Store backup
            storage_id = await self.storage.store(backup_id, backup_data, metadata)
            metadata.destination_path = storage_id
            
            # Calculate compression ratio
            original_size = sum(
                Path(f).stat().st_size 
                for f in files_to_backup 
                if Path(f).exists()
            )
            if original_size > 0:
                metadata.compression_ratio = (
                    (original_size - metadata.size_bytes) / original_size
                )
            
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.utcnow()
            
            self._backups[backup_id] = metadata
            
            # Create recovery point
            recovery_point = RecoveryPoint(
                recovery_point_id=str(uuid.uuid4()),
                backup_id=backup_id,
                timestamp=metadata.completed_at,
                description=f"{backup_type.value} backup"
            )
            self._recovery_points.append(recovery_point)
            
            logger.info(
                f"Backup {backup_id} completed: "
                f"{len(files_to_backup)} files, "
                f"{metadata.size_bytes} bytes"
            )
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            logger.error(f"Backup {backup_id} failed: {e}")
            raise
        
        return metadata
    
    def _collect_files(
        self,
        source_path: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[str]:
        """Collect files to backup."""
        source = Path(source_path)
        files = []
        
        if source.is_file():
            files = [str(source)]
        elif source.is_dir():
            files = [str(f) for f in source.rglob("*") if f.is_file()]
        
        # Apply include patterns
        if include_patterns:
            import fnmatch
            included = []
            for pattern in include_patterns:
                included.extend([
                    f for f in files 
                    if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)
                ])
            files = list(set(included))
        
        # Apply exclude patterns
        if exclude_patterns:
            import fnmatch
            for pattern in exclude_patterns:
                files = [
                    f for f in files 
                    if not (fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern))
                ]
        
        return files
    
    def _filter_incremental(
        self,
        files: List[str],
        since: datetime
    ) -> List[str]:
        """Filter files modified since a timestamp."""
        return [
            f for f in files
            if datetime.fromtimestamp(Path(f).stat().st_mtime) > since
        ]
    
    async def _create_archive(self, files: List[str]) -> bytes:
        """Create compressed archive of files."""
        import io
        import tarfile
        
        buffer = io.BytesIO()
        
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for file_path in files:
                path = Path(file_path)
                if path.exists():
                    tar.add(file_path, arcname=path.name)
        
        return buffer.getvalue()
    
    async def restore_backup(
        self,
        backup_id: str,
        destination_path: str,
        verify_checksum: bool = True
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup to restore
            destination_path: Where to restore
            verify_checksum: Whether to verify checksum
        """
        metadata = self._backups.get(backup_id)
        if not metadata:
            raise ValueError(f"Backup {backup_id} not found")

        # Allow restore from COMPLETED or VERIFIED backups
        if metadata.status not in (BackupStatus.COMPLETED, BackupStatus.VERIFIED):
            raise ValueError(f"Backup {backup_id} is not completed or verified (current status: {metadata.status.value})")

        logger.info({
            "msg": "Restoring backup",
            "backup_id": backup_id,
            "destination": destination_path,
            "status": metadata.status.value
        })
        
        # Retrieve backup data
        backup_data = await self.storage.retrieve(metadata.destination_path)
        
        # Verify checksum
        if verify_checksum:
            actual_checksum = hashlib.sha256(backup_data).hexdigest()
            if actual_checksum != metadata.checksum:
                raise ValueError("Checksum mismatch - backup may be corrupted")
        
        # Extract archive
        await self._extract_archive(backup_data, destination_path)
        
        logger.info(f"Restore of backup {backup_id} completed")
        return True
    
    async def _extract_archive(self, data: bytes, destination: str):
        """Extract archive to destination."""
        import io
        import tarfile
        
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        buffer = io.BytesIO(data)
        
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(path=destination)
    
    async def verify_backup(self, backup_id: str) -> bool:
        """Verify a backup's integrity."""
        metadata = self._backups.get(backup_id)
        if not metadata:
            return False
        
        try:
            backup_data = await self.storage.retrieve(metadata.destination_path)
            actual_checksum = hashlib.sha256(backup_data).hexdigest()
            
            if actual_checksum == metadata.checksum:
                metadata.status = BackupStatus.VERIFIED
                metadata.verified_at = datetime.utcnow()
                logger.info(f"Backup {backup_id} verified successfully")
                return True
            else:
                logger.error(f"Backup {backup_id} verification failed: checksum mismatch")
                return False
                
        except Exception as e:
            logger.error(f"Backup {backup_id} verification failed: {e}")
            return False
    
    def schedule_backup(
        self,
        schedule_id: str,
        source_path: str,
        cron_expression: str,  # Simplified: "daily", "weekly", or "hourly"
        backup_type: BackupType = BackupType.INCREMENTAL
    ):
        """Schedule recurring backups."""
        async def scheduled_task():
            while True:
                try:
                    await self.create_backup(source_path, backup_type)
                    
                    # Sleep based on schedule
                    if cron_expression == "hourly":
                        await asyncio.sleep(3600)
                    elif cron_expression == "daily":
                        await asyncio.sleep(86400)
                    elif cron_expression == "weekly":
                        await asyncio.sleep(604800)
                    else:
                        await asyncio.sleep(86400)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduled backup error: {e}")
                    await asyncio.sleep(3600)
        
        task = asyncio.create_task(scheduled_task())
        self._scheduled_tasks[schedule_id] = task
        
        logger.info(f"Scheduled backup {schedule_id}: {cron_expression}")
    
    def cancel_schedule(self, schedule_id: str):
        """Cancel a scheduled backup."""
        task = self._scheduled_tasks.pop(schedule_id, None)
        if task:
            task.cancel()
            logger.info(f"Cancelled scheduled backup {schedule_id}")
    
    def get_recovery_points(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[RecoveryPoint]:
        """Get available recovery points."""
        points = self._recovery_points
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return sorted(points, key=lambda p: p.timestamp, reverse=True)
    
    async def cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        
        to_delete = []
        for backup_id, metadata in self._backups.items():
            if metadata.completed_at and metadata.completed_at < cutoff:
                to_delete.append(backup_id)
        
        for backup_id in to_delete:
            metadata = self._backups[backup_id]
            await self.storage.delete(metadata.destination_path)
            del self._backups[backup_id]
            logger.info(f"Deleted old backup {backup_id}")
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_backups = len(self._backups)
        completed = sum(1 for b in self._backups.values() if b.status == BackupStatus.COMPLETED)
        failed = sum(1 for b in self._backups.values() if b.status == BackupStatus.FAILED)
        verified = sum(1 for b in self._backups.values() if b.status == BackupStatus.VERIFIED)
        
        total_size = sum(b.size_bytes for b in self._backups.values())
        
        return {
            "total_backups": total_backups,
            "completed": completed,
            "failed": failed,
            "verified": verified,
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "recovery_points": len(self._recovery_points),
            "scheduled_tasks": len(self._scheduled_tasks),
            "retention_days": self.retention_days
        }


class DisasterRecovery:
    """
    Disaster recovery procedures.
    """
    
    def __init__(self, backup_engine: BackupEngine):
        self.backup_engine = backup_engine
        self._dr_site: Optional[str] = None
        self._replication_enabled = False
    
    def configure_dr_site(self, site_url: str):
        """Configure disaster recovery site."""
        self._dr_site = site_url
        logger.info(f"DR site configured: {site_url}")
    
    async def failover(self, backup_id: str, target_path: str) -> bool:
        """
        Execute failover to backup.
        
        Args:
            backup_id: Backup to failover to
            target_path: Target for restoration
        """
        logger.info(f"Executing failover to backup {backup_id}")
        
        try:
            await self.backup_engine.restore_backup(backup_id, target_path)
            logger.info(f"Failover to backup {backup_id} completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False
    
    async def test_recovery(
        self,
        backup_id: str,
        test_path: str
    ) -> Dict[str, Any]:
        """
        Test recovery procedure without affecting production.
        
        Returns:
            Test results
        """
        logger.info(f"Testing recovery of backup {backup_id}")
        
        start_time = datetime.utcnow()
        
        try:
            await self.backup_engine.restore_backup(backup_id, test_path)
            
            # Verify restored data
            restored_files = list(Path(test_path).rglob("*"))
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "success": True,
                "backup_id": backup_id,
                "test_path": test_path,
                "duration_seconds": duration,
                "restored_files": len(restored_files),
                "verified_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "backup_id": backup_id,
                "error": str(e),
                "verified_at": datetime.utcnow().isoformat()
            }
    
    def generate_dr_plan(self) -> Dict[str, Any]:
        """Generate disaster recovery plan."""
        stats = self.backup_engine.get_backup_stats()
        
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "rpo_hours": 24,  # Recovery Point Objective
            "rto_hours": 4,   # Recovery Time Objective
            "backup_stats": stats,
            "recovery_points": [
                {
                    "id": rp.recovery_point_id,
                    "timestamp": rp.timestamp.isoformat(),
                    "backup_id": rp.backup_id
                }
                for rp in self.backup_engine.get_recovery_points()[:10]
            ],
            "procedures": {
                "full_restore": [
                    "1. Identify backup to restore from",
                    "2. Verify backup integrity",
                    "3. Stop application services",
                    "4. Execute restore operation",
                    "5. Verify restored data",
                    "6. Restart application services"
                ],
                "point_in_time_recovery": [
                    "1. Identify target recovery point",
                    "2. Find closest backup before target time",
                    "3. Restore full backup",
                    "4. Apply incremental backups",
                    "5. Verify data at target time"
                ]
            }
        }


__all__ = [
    "BackupEngine",
    "BackupStorage",
    "LocalBackupStorage",
    "S3BackupStorage",
    "GCSBackupStorage",
    "AzureBackupStorage",
    "create_storage_backend",
    "BackupMetadata",
    "BackupType",
    "BackupStatus",
    "RecoveryPoint",
    "DisasterRecovery"
]
