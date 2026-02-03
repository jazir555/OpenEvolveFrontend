"""
Unified Credential Management System

Manages LLM credentials from multiple sources:
1. OpenEvolve configuration (environment variables, config files)
2. BubbleLab credentials API (saved credentials)
3. User-provided credentials (runtime)

All credentials are verified and tracked for usage.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path
import json
import httpx

from .team_assignment import (
    LLMCredential,
    LLMProvider,
    CredentialSource,
    CredentialVerificationRequest,
    CredentialVerificationResponse,
)
import structlog

logger = structlog.get_logger()


class CredentialManager:
    """
    Unified credential manager for LLM APIs

    Pulls credentials from:
    1. OpenEvolve config (env vars, .env files)
    2. BubbleLab credentials API
    3. Runtime user input
    """

    def __init__(self, bubblelab_api_url: Optional[str] = None):
        """
        Initialize credential manager

        Args:
            bubblelab_api_url: URL for BubbleLab API (for credential fetching)
        """
        self.bubblelab_api_url = bubblelab_api_url or os.getenv(
            "BUBBLELAB_API_URL",
            "http://localhost:3001"
        )
        self._credentials_cache: Dict[str, LLMCredential] = {}
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Close HTTP client"""
        await self._client.aclose()

    # ==================== Credential Loading ====================

    async def load_all_credentials(self) -> List[LLMCredential]:
        """
        Load credentials from all sources

        Returns:
            List of all available credentials
        """
        logger.info("loading_credentials_from_all_sources")

        credentials = []

        # 1. Load from OpenEvolve config
        config_creds = await self._load_from_openevolve_config()
        credentials.extend(config_creds)
        logger.info("loaded_credentials_from_config", count=len(config_creds))

        # 2. Load from BubbleLab credentials API
        bubblelab_creds = await self._load_from_bubblelab_credentials()
        credentials.extend(bubblelab_creds)
        logger.info("loaded_credentials_from_bubblelab", count=len(bubblelab_creds))

        # Cache credentials
        for cred in credentials:
            self._credentials_cache[cred.credential_id] = cred

        logger.info("total_credentials_loaded", total=len(credentials))

        return credentials

    async def _load_from_openevolve_config(self) -> List[LLMCredential]:
        """
        Load credentials from OpenEvolve environment/config

        Environment variables:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY
        - OPENROUTER_API_KEY
        - GROQ_API_KEY
        - DEEPSEEK_API_KEY
        - CUSTOM_LLMS (JSON with custom credentials)
        """
        credentials = []

        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_openai",
                provider=LLMProvider.OPENAI,
                api_key=openai_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,  # Will verify on first use
                model_permissions=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            ))

        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_anthropic",
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,
                model_permissions=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            ))

        # Google
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_google",
                provider=LLMProvider.GOOGLE,
                api_key=google_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,
                model_permissions=["gemini-pro", "gemini-ultra"],
                project_id=os.getenv("GOOGLE_PROJECT_ID"),
                region=os.getenv("GOOGLE_REGION", "us-central1"),
            ))

        # OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_openrouter",
                provider=LLMProvider.OPENROUTER,
                api_key=openrouter_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,
                model_permissions=["*"],  # OpenRouter can access many models
            ))

        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_groq",
                provider=LLMProvider.GROQ,
                api_key=groq_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,
                model_permissions=["llama3-70b-8192", "llama3-8b-8192"],
            ))

        # DeepSeek
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            credentials.append(LLMCredential(
                credential_id="openevolve_deepseek",
                provider=LLMProvider.DEEPSEEK,
                api_key=deepseek_key,
                source=CredentialSource.OPENEVOLVE_CONFIG,
                verified=False,
                model_permissions=["deepseek-coder", "deepseek-chat"],
            ))

        # Custom LLMs (from JSON env var)
        custom_llms_json = os.getenv("CUSTOM_LLMS")
        if custom_llms_json:
            try:
                custom_llms = json.loads(custom_llms_json)
                for custom in custom_llms:
                    credentials.append(LLMCredential(
                        credential_id=f"openevolve_custom_{custom['name']}",
                        provider=LLMProvider.OPENAI_LIKE,
                        api_key=custom["api_key"],
                        source=CredentialSource.OPENEVOLVE_CONFIG,
                        verified=False,
                        model_permissions=custom.get("models", ["*"]),
                        api_base=custom.get("api_base"),
                    ))
            except Exception as e:
                logger.error("failed_to_parse_custom_llms", error=str(e))

        return credentials

    async def _load_from_bubblelab_credentials(self) -> List[LLMCredential]:
        """
        Load credentials from BubbleLab credentials API

        Calls: GET /api/credentials
        Returns saved credentials with metadata
        """
        credentials = []

        try:
            response = await self._client.get(
                f"{self.bubblelab_api_url}/api/credentials"
            )

            if response.status_code == 200:
                data = response.json()

                for cred_data in data.get("credentials", []):
                    # Convert BubbleLab credential to LLMCredential
                    credential = LLMCredential(
                        credential_id=f"bubblelab_{cred_data['id']}",
                        provider=LLMProvider(cred_data["provider"]),
                        api_key=cred_data["api_key"],
                        source=CredentialSource.BUBBLELAB_CREDENTIALS,
                        verified=cred_data.get("verified", False),
                        verified_at=cred_data.get("verified_at"),
                        last_used=cred_data.get("last_used"),
                        model_permissions=cred_data.get("models", []),
                        api_base=cred_data.get("api_base"),
                    )
                    credentials.append(credential)

                logger.info("loaded_bubblelab_credentials", count=len(credentials))

        except Exception as e:
            logger.warning(
                "failed_to_load_bubblelab_credentials",
                error=str(e),
                note="Credentials might not be available yet"
            )

        return credentials

    # ==================== Credential Retrieval ====================

    async def get_credential(
        self,
        provider: LLMProvider,
        model: Optional[str] = None,
    ) -> Optional[LLMCredential]:
        """
        Get best credential for a provider/model

        Args:
            provider: LLM provider
            model: Optional specific model

        Returns:
            Best matching credential or None
        """
        # Reload cache if empty
        if not self._credentials_cache:
            await self.load_all_credentials()

        # Find credentials for provider
        provider_creds = [
            cred for cred in self._credentials_cache.values()
            if cred.provider == provider
        ]

        if not provider_creds:
            logger.warning("no_credentials_found", provider=provider.value)
            return None

        # If specific model requested, find credential that can access it
        if model:
            for cred in provider_creds:
                if model in cred.model_permissions or "*" in cred.model_permissions:
                    return cred

        # Return first available credential (prefer verified)
        verified_creds = [c for c in provider_creds if c.verified]
        if verified_creds:
            return verified_creds[0]

        return provider_creds[0]

    async def get_all_credentials(self) -> List[LLMCredential]:
        """Get all loaded credentials"""
        if not self._credentials_cache:
            await self.load_all_credentials()

        return list(self._credentials_cache.values())

    # ==================== Credential Verification ====================

    async def verify_credential(
        self,
        request: CredentialVerificationRequest,
    ) -> CredentialVerificationResponse:
        """
        Verify an LLM credential by making a test API call

        Args:
            request: Verification request

        Returns:
            Verification result
        """
        logger.info(
            "verifying_credential",
            provider=request.provider.value,
            has_api_base=request.api_base is not None,
        )

        start_time = datetime.now(timezone.utc)

        try:
            if request.provider == LLMProvider.OPENAI:
                return await self._verify_openai(request, start_time)
            elif request.provider == LLMProvider.ANTHROPIC:
                return await self._verify_anthropic(request, start_time)
            elif request.provider == LLMProvider.GOOGLE:
                return await self._verify_google(request, start_time)
            elif request.provider == LLMProvider.OPENAI_LIKE:
                return await self._verify_openai_like(request, start_time)
            else:
                return CredentialVerificationResponse(
                    verified=False,
                    message=f"Verification not implemented for {request.provider.value}",
                )

        except Exception as e:
            logger.error("credential_verification_failed", error=str(e))
            return CredentialVerificationResponse(
                verified=False,
                message=f"Verification failed: {str(e)}",
            )

    async def _verify_openai(
        self,
        request: CredentialVerificationRequest,
        start_time: datetime,
    ) -> CredentialVerificationResponse:
        """Verify OpenAI credential"""
        api_base = request.api_base or "https://api.openai.com/v1"
        model = request.model_to_test or "gpt-3.5-turbo"

        response = await self._client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {request.api_key}",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
            timeout=10.0,
        )

        latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if response.status_code == 200:
            # Get available models
            models_response = await self._client.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {request.api_key}"},
            )

            available_models = []
            if models_response.status_code == 200:
                models_data = models_response.json()
                available_models = [
                    m["id"] for m in models_data.get("data", [])
                    if m.get("owned_by", "organization") == "organization"
                ]

            return CredentialVerificationResponse(
                verified=True,
                message="Credential verified successfully",
                test_model=model,
                latency_ms=latency,
                available_models=available_models,
            )
        else:
            return CredentialVerificationResponse(
                verified=False,
                message=f"Verification failed: {response.status_code}",
                latency_ms=latency,
            )

    async def _verify_anthropic(
        self,
        request: CredentialVerificationRequest,
        start_time: datetime,
    ) -> CredentialVerificationResponse:
        """Verify Anthropic credential"""
        model = request.model_to_test or "claude-3-haiku-20240307"

        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": request.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "test"}],
            },
            timeout=10.0,
        )

        latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if response.status_code == 200:
            return CredentialVerificationResponse(
                verified=True,
                message="Credential verified successfully",
                test_model=model,
                latency_ms=latency,
                available_models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            )
        else:
            return CredentialVerificationResponse(
                verified=False,
                message=f"Verification failed: {response.status_code}",
                latency_ms=latency,
            )

    async def _verify_google(
        self,
        request: CredentialVerificationRequest,
        start_time: datetime,
    ) -> CredentialVerificationResponse:
        """Verify Google credential"""
        # Simplified Google verification
        model = request.model_to_test or "gemini-pro"

        response = await self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={request.api_key}",
            json={
                "contents": [{"parts": [{"text": "test"}]}],
            },
            timeout=10.0,
        )

        latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if response.status_code == 200:
            return CredentialVerificationResponse(
                verified=True,
                message="Credential verified successfully",
                test_model=model,
                latency_ms=latency,
                available_models=["gemini-pro", "gemini-pro-vision"],
            )
        else:
            return CredentialVerificationResponse(
                verified=False,
                message=f"Verification failed: {response.status_code}",
                latency_ms=latency,
            )

    async def _verify_openai_like(
        self,
        request: CredentialVerificationRequest,
        start_time: datetime,
    ) -> CredentialVerificationResponse:
        """Verify OpenAI-compatible API (vLLM, Ollama, etc.)"""
        if not request.api_base:
            return CredentialVerificationResponse(
                verified=False,
                message="api_base required for openai-like providers",
            )

        model = request.model_to_test or "default"

        response = await self._client.post(
            f"{request.api_base.rstrip('/')}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {request.api_key}",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5,
            },
            timeout=10.0,
        )

        latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        if response.status_code == 200:
            return CredentialVerificationResponse(
                verified=True,
                message="OpenAI-compatible API verified",
                test_model=model,
                latency_ms=latency,
            )
        else:
            return CredentialVerificationResponse(
                verified=False,
                message=f"Verification failed: {response.status_code}",
                latency_ms=latency,
            )

    # ==================== Credential Saving ====================

    async def save_credential(
        self,
        credential: LLMCredential,
        save_to_bubblelab: bool = True,
    ) -> LLMCredential:
        """
        Save a credential (to BubbleLab credentials API)

        Args:
            credential: Credential to save
            save_to_bubblelab: Whether to save to BubbleLab (vs just cache)

        Returns:
            Saved credential with ID
        """
        logger.info(
            "saving_credential",
            provider=credential.provider.value,
            source=credential.source.value,
        )

        if save_to_bubblelab and credential.source == CredentialSource.USER_PROVIDED:
            # Save to BubbleLab credentials API
            try:
                response = await self._client.post(
                    f"{self.bubblelab_api_url}/api/credentials",
                    json={
                        "provider": credential.provider.value,
                        "api_key": credential.api_key,
                        "models": credential.model_permissions,
                        "api_base": credential.api_base,
                        "verified": credential.verified,
                        "verified_at": credential.verified_at,
                    },
                )

                if response.status_code == 201:
                    data = response.json()
                    credential_id = f"bubblelab_{data['id']}"
                    credential.credential_id = credential_id
                    logger.info("credential_saved_to_bubblelab", credential_id=credential_id)

            except Exception as e:
                logger.warning(
                    "failed_to_save_to_bubblelab",
                    error=str(e),
                    note="Credential will be cached only"
                )

        # Cache credential
        self._credentials_cache[credential.credential_id] = credential

        return credential


# Singleton instance
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get or create credential manager singleton"""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager
