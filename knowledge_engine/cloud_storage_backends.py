"""
Cloud Storage Backends for Backup and Recovery

Provides concrete implementations of backup storage for:
- AWS S3
- Google Cloud Storage (GCS)
- Azure Blob Storage
- SFTP/SSH

All implementations follow the BackupStorage interface from backup_recovery.py.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class StorageCredentials:
    """Base class for storage credentials."""
    pass


@dataclass
class S3Credentials(StorageCredentials):
    """AWS S3 credentials."""
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None  # For MinIO compatibility
    
    @classmethod
    def from_env(cls) -> S3Credentials:
        """Create credentials from environment variables.

        Raises:
            ValueError: If required credentials are not set
        """
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not access_key or not secret_key:
            raise ValueError(
                "AWS credentials not found in environment. "
                "Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
            )

        return cls(
            access_key_id=access_key,
            secret_access_key=secret_key,
            region=os.getenv("AWS_REGION", "us-east-1"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL")
        )


@dataclass
class GCSCredentials(StorageCredentials):
    """Google Cloud Storage credentials."""
    project_id: str
    credentials_path: Optional[str] = None  # Path to service account JSON
    credentials_json: Optional[str] = None  # Raw JSON string
    
    @classmethod
    def from_env(cls) -> GCSCredentials:
        """Create credentials from environment variables.

        Raises:
            ValueError: If required credentials are not set
        """
        project_id = os.getenv("GCS_PROJECT_ID")

        if not project_id:
            raise ValueError(
                "GCS project ID not found in environment. "
                "Required: GCS_PROJECT_ID"
            )

        return cls(
            project_id=project_id,
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            credentials_json=os.getenv("GCS_CREDENTIALS_JSON")
        )


@dataclass
class AzureCredentials(StorageCredentials):
    """Azure Blob Storage credentials."""
    account_name: str
    account_key: Optional[str] = None
    connection_string: Optional[str] = None
    sas_token: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> AzureCredentials:
        """Create credentials from environment variables.

        Raises:
            ValueError: If required credentials are not set
        """
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT")

        if not account_name:
            raise ValueError(
                "Azure storage account not found in environment. "
                "Required: AZURE_STORAGE_ACCOUNT"
            )

        return cls(
            account_name=account_name,
            account_key=os.getenv("AZURE_STORAGE_KEY"),
            connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
            sas_token=os.getenv("AZURE_STORAGE_SAS_TOKEN")
        )


@dataclass
class SFTPCredentials(StorageCredentials):
    """SFTP/SSH credentials."""
    host: str
    port: int = 22
    username: str = ""
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> SFTPCredentials:
        """Create credentials from environment variables.

        Raises:
            ValueError: If required credentials are not set
        """
        host = os.getenv("SFTP_HOST")

        if not host:
            raise ValueError(
                "SFTP host not found in environment. "
                "Required: SFTP_HOST"
            )

        return cls(
            host=host,
            port=int(os.getenv("SFTP_PORT", "22")),
            username=os.getenv("SFTP_USERNAME", ""),
            password=os.getenv("SFTP_PASSWORD"),
            private_key_path=os.getenv("SFTP_PRIVATE_KEY_PATH"),
            private_key_passphrase=os.getenv("SFTP_KEY_PASSPHRASE")
        )


class S3BackupStorage:
    """AWS S3 backup storage implementation."""
    
    def __init__(
        self,
        bucket_name: str,
        credentials: Optional[S3Credentials] = None,
        prefix: str = "backups/",
        storage_class: str = "STANDARD"
    ):
        self.bucket_name = bucket_name
        self.credentials = credentials or S3Credentials.from_env()
        self.prefix = prefix
        self.storage_class = storage_class
        self._s3_client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize S3 client."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            session = boto3.Session(
                aws_access_key_id=self.credentials.access_key_id,
                aws_secret_access_key=self.credentials.secret_access_key,
                region_name=self.credentials.region
            )
            
            kwargs = {}
            if self.credentials.endpoint_url:
                kwargs['endpoint_url'] = self.credentials.endpoint_url
            
            self._s3_client = session.client('s3', **kwargs)
            self._client_error = ClientError
            
            # Verify bucket exists
            self._s3_client.head_bucket(Bucket=self.bucket_name)
            
            logger.info({
                "msg": "S3 storage initialized",
                "bucket": self.bucket_name,
                "region": self.credentials.region
            })
            
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install with: pip install boto3"
            )
        except Exception as e:
            logger.error({"msg": "Failed to initialize S3 client", "error": str(e)})
            raise
    
    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: Any
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
                "size": len(data)
            })
            
            return storage_id
            
        except self._client_error as e:
            logger.error({"msg": "S3 store failed", "error": str(e)})
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
                "size": len(data)
            })
            
            return data
            
        except self._client_error as e:
            logger.error({"msg": "S3 retrieve failed", "error": str(e)})
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
            
            logger.info({"msg": "Backup deleted from S3", "key": key})
            return True
            
        except self._client_error as e:
            logger.error({"msg": "S3 delete failed", "error": str(e)})
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
                    backups.append(backup_id)
            
            return backups
            
        except self._client_error as e:
            logger.error({"msg": "S3 list failed", "error": str(e)})
            return []


class GCSBackupStorage:
    """Google Cloud Storage backup storage implementation."""
    
    def __init__(
        self,
        bucket_name: str,
        credentials: Optional[GCSCredentials] = None,
        prefix: str = "backups/"
    ):
        self.bucket_name = bucket_name
        self.credentials = credentials or GCSCredentials.from_env()
        self.prefix = prefix
        self._client = None
        self._bucket = None
        self._init_client()
    
    def _init_client(self):
        """Initialize GCS client."""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            
            if self.credentials.credentials_path:
                # Load from service account file
                creds = service_account.Credentials.from_service_account_file(
                    self.credentials.credentials_path
                )
                self._client = storage.Client(
                    project=self.credentials.project_id,
                    credentials=creds
                )
            elif self.credentials.credentials_json:
                # Load from JSON string
                import json
                creds_info = json.loads(self.credentials.credentials_json)
                creds = service_account.Credentials.from_service_account_info(
                    creds_info
                )
                self._client = storage.Client(
                    project=self.credentials.project_id,
                    credentials=creds
                )
            else:
                # Use default credentials
                self._client = storage.Client(project=self.credentials.project_id)
            
            self._bucket = self._client.bucket(self.bucket_name)
            
            logger.info({
                "msg": "GCS storage initialized",
                "bucket": self.bucket_name,
                "project": self.credentials.project_id
            })
            
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required. Install with: pip install google-cloud-storage"
            )
        except Exception as e:
            logger.error({"msg": "Failed to initialize GCS client", "error": str(e)})
            raise
    
    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: Any
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
                "key": key
            })
            
            return storage_id
            
        except Exception as e:
            logger.error({"msg": "GCS store failed", "error": str(e)})
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
                "size": len(data)
            })
            
            return data
            
        except Exception as e:
            logger.error({"msg": "GCS retrieve failed", "error": str(e)})
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
            
            logger.info({"msg": "Backup deleted from GCS", "key": key})
            return True
            
        except Exception as e:
            logger.error({"msg": "GCS delete failed", "error": str(e)})
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
                    backup_id = blob.name[len(self.prefix):].replace('/data.gz', '')
                    backups.append(backup_id)
            
            return backups
            
        except Exception as e:
            logger.error({"msg": "GCS list failed", "error": str(e)})
            return []


class AzureBackupStorage:
    """Azure Blob Storage backup storage implementation."""
    
    def __init__(
        self,
        container_name: str,
        credentials: Optional[AzureCredentials] = None,
        prefix: str = "backups/"
    ):
        self.container_name = container_name
        self.credentials = credentials or AzureCredentials.from_env()
        self.prefix = prefix
        self._container_client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Azure Blob client."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            if self.credentials.connection_string:
                self._blob_service = BlobServiceClient.from_connection_string(
                    self.credentials.connection_string
                )
            elif self.credentials.account_key:
                account_url = f"https://{self.credentials.account_name}.blob.core.windows.net"
                self._blob_service = BlobServiceClient(
                    account_url=account_url,
                    credential=self.credentials.account_key
                )
            else:
                # Use default Azure credentials
                from azure.identity import DefaultAzureCredential
                account_url = f"https://{self.credentials.account_name}.blob.core.windows.net"
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
                "account": self.credentials.account_name
            })
            
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required. Install with: pip install azure-storage-blob"
            )
        except Exception as e:
            logger.error({"msg": "Failed to initialize Azure client", "error": str(e)})
            raise
    
    async def store(
        self,
        backup_id: str,
        data: bytes,
        metadata: Any
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
                "key": key
            })
            
            return storage_id
            
        except Exception as e:
            logger.error({"msg": "Azure store failed", "error": str(e)})
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
                "size": len(data)
            })
            
            return data
            
        except Exception as e:
            logger.error({"msg": "Azure retrieve failed", "error": str(e)})
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
            
            logger.info({"msg": "Backup deleted from Azure", "key": key})
            return True
            
        except Exception as e:
            logger.error({"msg": "Azure delete failed", "error": str(e)})
            return False
    
    async def list_backups(self) -> List[str]:
        """List available backups in Azure."""
        try:
            blobs = self._container_client.list_blobs(name_starts_with=self.prefix)
            
            backups = []
            for blob in blobs:
                if blob.name.endswith('/data.gz'):
                    backup_id = blob.name[len(self.prefix):].replace('/data.gz', '')
                    backups.append(backup_id)
            
            return backups
            
        except Exception as e:
            logger.error({"msg": "Azure list failed", "error": str(e)})
            return []


def create_cloud_storage(
    storage_type: str,
    bucket_or_container: str,
    **kwargs
):
    """
    Factory function to create cloud storage backend.
    
    Args:
        storage_type: 's3', 'gcs', or 'azure'
        bucket_or_container: Bucket name (S3/GCS) or container name (Azure)
        **kwargs: Additional configuration options
        
    Returns:
        Cloud storage backend instance
    """
    storage_type = storage_type.lower()
    
    if storage_type == 's3':
        credentials = kwargs.get('credentials') or S3Credentials.from_env()
        return S3BackupStorage(
            bucket_name=bucket_or_container,
            credentials=credentials,
            prefix=kwargs.get('prefix', 'backups/'),
            storage_class=kwargs.get('storage_class', 'STANDARD')
        )
    
    elif storage_type == 'gcs':
        credentials = kwargs.get('credentials') or GCSCredentials.from_env()
        return GCSBackupStorage(
            bucket_name=bucket_or_container,
            credentials=credentials,
            prefix=kwargs.get('prefix', 'backups/')
        )
    
    elif storage_type == 'azure':
        credentials = kwargs.get('credentials') or AzureCredentials.from_env()
        return AzureBackupStorage(
            container_name=bucket_or_container,
            credentials=credentials,
            prefix=kwargs.get('prefix', 'backups/')
        )
    
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")


# Export all classes
__all__ = [
    'S3BackupStorage',
    'GCSBackupStorage',
    'AzureBackupStorage',
    'S3Credentials',
    'GCSCredentials',
    'AzureCredentials',
    'SFTPCredentials',
    'create_cloud_storage'
]
