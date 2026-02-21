#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Model settings API routes"""

import io
import threading

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from opencontext.config.global_config import GlobalConfig
from opencontext.llm.global_embedding_client import GlobalEmbeddingClient
from opencontext.llm.global_vlm_client import GlobalVLMClient
from opencontext.llm.llm_client import LLMClient, LLMType
from opencontext.server.middleware.auth import auth_dependency
from opencontext.server.utils import convert_resp
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["model-settings"])
_config_lock = threading.Lock()


# ==================== Data Models ====================


class ModelSettingsVO(BaseModel):
    """Model settings with optional separate embedding configuration"""

    modelPlatform: str
    modelId: str
    baseUrl: str
    apiKey: str
    embeddingModelId: str
    embeddingBaseUrl: str | None = None
    embeddingApiKey: str | None = None
    embeddingModelPlatform: str | None = None


class GetModelSettingsResponse(BaseModel):
    config: ModelSettingsVO


class UpdateModelSettingsRequest(BaseModel):
    config: ModelSettingsVO


class UpdateModelSettingsResponse(BaseModel):
    success: bool
    message: str


# ==================== Helper Functions ====================


def _build_llm_config(
    base_url: str, api_key: str, model: str, provider: str, llm_type: LLMType, **kwargs
) -> dict:
    """Build LLM config dict"""
    config = {"base_url": base_url, "api_key": api_key, "model": model, "provider": provider}

    # Add optional parameters
    if "timeout" in kwargs:
        config["timeout"] = kwargs["timeout"]

    if llm_type == LLMType.EMBEDDING:
        config["output_dim"] = kwargs.get("output_dim", 2048)
    return config


# ==================== API Endpoints ====================


@router.get("/api/model_settings/get")
async def get_model_settings(_auth: str = auth_dependency):
    """Get current model configuration"""
    try:
        config = GlobalConfig.get_instance().get_config()
        if not config:
            return convert_resp(code=500, status=500, message="配置未初始化")

        vlm_cfg = config.get("vlm_model", {})
        emb_cfg = config.get("embedding_model", {})

        settings = ModelSettingsVO(
            modelPlatform=vlm_cfg.get("provider", ""),
            modelId=vlm_cfg.get("model", ""),
            baseUrl=vlm_cfg.get("base_url", ""),
            apiKey=vlm_cfg.get("api_key", ""),
            embeddingModelId=emb_cfg.get("model", ""),
            embeddingBaseUrl=emb_cfg.get("base_url", ""),
            embeddingApiKey=emb_cfg.get("api_key", ""),
            embeddingModelPlatform=emb_cfg.get("provider", ""),
        )

        return convert_resp(data=GetModelSettingsResponse(config=settings).model_dump())

    except Exception as e:
        logger.exception(f"Failed to get model settings: {e}")
        return convert_resp(code=500, status=500, message=f"获取模型设置失败: {str(e)}")


@router.post("/api/model_settings/update")
async def update_model_settings(request: UpdateModelSettingsRequest, _auth: str = auth_dependency):
    """Update model configuration and reinitialize LLM clients"""
    with _config_lock:
        try:
            cfg = request.config

            # Use API keys directly from frontend
            vlm_key = cfg.apiKey
            emb_key = cfg.embeddingApiKey or vlm_key

            # Resolve embedding URL and provider
            emb_url = cfg.embeddingBaseUrl or cfg.baseUrl
            emb_provider = cfg.embeddingModelPlatform or cfg.modelPlatform

            # Validation
            if not vlm_key:
                return convert_resp(code=400, status=400, message="VLM API key cannot be empty")
            if not emb_key:
                return convert_resp(
                    code=400, status=400, message="Embedding API key cannot be empty"
                )
            if not cfg.modelId:
                return convert_resp(code=400, status=400, message="VLM model ID cannot be empty")
            if not cfg.embeddingModelId:
                return convert_resp(
                    code=400, status=400, message="Embedding model ID cannot be empty"
                )

            # Validate VLM
            vlm_config = _build_llm_config(
                cfg.baseUrl, vlm_key, cfg.modelId, cfg.modelPlatform, LLMType.CHAT, timeout=15
            )
            vlm_valid, vlm_msg = LLMClient(llm_type=LLMType.CHAT, config=vlm_config).validate()
            if not vlm_valid:
                return convert_resp(
                    code=400, status=400, message=f"VLM validation failed: {vlm_msg}"
                )

            # Validate Embedding
            emb_config = _build_llm_config(
                emb_url, emb_key, cfg.embeddingModelId, emb_provider, LLMType.EMBEDDING, timeout=15
            )
            emb_valid, emb_msg = LLMClient(llm_type=LLMType.EMBEDDING, config=emb_config).validate()
            if not emb_valid:
                return convert_resp(
                    code=400, status=400, message=f"Embedding validation failed: {emb_msg}"
                )

            # Save configuration (without timeout limit)
            vlm_config_save = _build_llm_config(
                cfg.baseUrl, vlm_key, cfg.modelId, cfg.modelPlatform, LLMType.CHAT
            )
            emb_config_save = _build_llm_config(
                emb_url, emb_key, cfg.embeddingModelId, emb_provider, LLMType.EMBEDDING
            )

            new_settings = {"vlm_model": vlm_config_save, "embedding_model": emb_config_save}

            config_mgr = GlobalConfig.get_instance().get_config_manager()
            if not config_mgr:
                return convert_resp(code=500, status=500, message="Config manager not initialized")

            if not config_mgr.save_user_settings(new_settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

            config_mgr.load_config(config_mgr.get_config_path())

            # Reinitialize clients
            if not GlobalVLMClient.get_instance().reinitialize():
                return convert_resp(
                    code=500, status=500, message="Failed to reinitialize VLM client"
                )
            if not GlobalEmbeddingClient.get_instance().reinitialize():
                return convert_resp(
                    code=500, status=500, message="Failed to reinitialize embedding client"
                )

            logger.info("Model settings updated successfully")
            return convert_resp(
                data=UpdateModelSettingsResponse(
                    success=True, message="Model settings updated successfully"
                ).model_dump()
            )

        except Exception as e:
            logger.exception(f"Failed to update model settings: {e}")
            return convert_resp(code=500, status=500, message="Failed to update model settings")


@router.post("/api/model_settings/validate")
async def validate_llm_config(request: UpdateModelSettingsRequest, _auth: str = auth_dependency):
    """Validate LLM configuration from frontend (without saving)"""
    try:
        cfg = request.config

        # Use API keys directly from frontend
        vlm_key = cfg.apiKey
        emb_key = cfg.embeddingApiKey or vlm_key

        # Resolve embedding URL and provider
        emb_url = cfg.embeddingBaseUrl or cfg.baseUrl
        emb_provider = cfg.embeddingModelPlatform or cfg.modelPlatform

        # Validation
        if not vlm_key:
            return convert_resp(code=400, status=400, message="VLM API key cannot be empty")
        if not emb_key:
            return convert_resp(code=400, status=400, message="Embedding API key cannot be empty")
        if not cfg.modelId:
            return convert_resp(code=400, status=400, message="VLM model ID cannot be empty")
        if not cfg.embeddingModelId:
            return convert_resp(code=400, status=400, message="Embedding model ID cannot be empty")

        # Build configs for validation (without saving)
        vlm_config = _build_llm_config(
            cfg.baseUrl, vlm_key, cfg.modelId, cfg.modelPlatform, LLMType.CHAT, timeout=15
        )
        emb_config = _build_llm_config(
            emb_url, emb_key, cfg.embeddingModelId, emb_provider, LLMType.EMBEDDING, timeout=15
        )

        # Validate VLM
        vlm_valid, vlm_msg = LLMClient(llm_type=LLMType.CHAT, config=vlm_config).validate()

        # Validate Embedding
        emb_valid, emb_msg = LLMClient(llm_type=LLMType.EMBEDDING, config=emb_config).validate()

        # Build error message
        if not vlm_valid or not emb_valid:
            errors = []
            if not vlm_valid:
                errors.append(f"VLM: {vlm_msg}")
            if not emb_valid:
                errors.append(f"Embedding: {emb_msg}")
            error_msg = "; ".join(errors)
            return convert_resp(code=400, status=400, message=error_msg)

        return convert_resp(code=0, status=200, message="连接测试成功！VLM和Embedding模型均正常")

    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return convert_resp(code=500, status=500, message=f"Validation failed: {str(e)}")


# ==================== General Settings ====================


class GeneralSettingsRequest(BaseModel):
    """General system settings request"""

    capture: dict | None = None
    processing: dict | None = None
    logging: dict | None = None


@router.get("/api/settings/general")
async def get_general_settings(_auth: str = auth_dependency):
    """Get general system settings"""
    try:
        config = GlobalConfig.get_instance().get_config()
        if not config:
            return convert_resp(code=500, status=500, message="Configuration not initialized")

        settings = {
            "capture": config.get("capture", {}),
            "processing": config.get("processing", {}),
            "logging": config.get("logging", {}),
        }

        return convert_resp(data=settings)

    except Exception as e:
        logger.exception(f"Failed to get general settings: {e}")
        return convert_resp(
            code=500, status=500, message=f"Failed to get general settings: {str(e)}"
        )


@router.post("/api/settings/general")
async def update_general_settings(request: GeneralSettingsRequest, _auth: str = auth_dependency):
    """Update general system settings"""
    with _config_lock:
        try:
            config_mgr = GlobalConfig.get_instance().get_config_manager()
            if not config_mgr:
                return convert_resp(code=500, status=500, message="Config manager not initialized")

            # Build settings dict
            settings = {}
            if request.capture is not None:
                settings["capture"] = request.capture
            if request.processing is not None:
                settings["processing"] = request.processing
            if request.logging is not None:
                settings["logging"] = request.logging

            if not settings:
                return convert_resp(code=400, status=400, message="No settings provided")

            # Save to user_setting.yaml
            if not config_mgr.save_user_settings(settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

            # Reload config
            config_mgr.load_config(config_mgr.get_config_path())

            logger.info("General settings updated successfully")
            return convert_resp(code=0, status=200, message="Settings updated successfully")

        except Exception as e:
            logger.exception(f"Failed to update general settings: {e}")
            return convert_resp(
                code=500, status=500, message=f"Failed to update settings: {str(e)}"
            )


# ==================== Prompts Settings ====================


class PromptsUpdateRequest(BaseModel):
    """Prompts update request"""

    prompts: dict


@router.get("/api/settings/prompts")
async def get_prompts(_auth: str = auth_dependency):
    """Get current prompts"""
    try:
        prompt_mgr = GlobalConfig.get_instance().get_prompt_manager()
        if not prompt_mgr:
            return convert_resp(code=500, status=500, message="Prompt manager not initialized")

        return convert_resp(data={"prompts": prompt_mgr.prompts})

    except Exception as e:
        logger.exception(f"Failed to get prompts: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to get prompts: {str(e)}")


@router.post("/api/settings/prompts")
async def update_prompts(request: PromptsUpdateRequest, _auth: str = auth_dependency):
    """Update prompts"""
    try:
        prompt_mgr = GlobalConfig.get_instance().get_prompt_manager()
        if not prompt_mgr:
            return convert_resp(code=500, status=500, message="Prompt manager not initialized")

        if not prompt_mgr.save_prompts(request.prompts):
            return convert_resp(code=500, status=500, message="Failed to save prompts")

        logger.info("Prompts updated successfully")
        return convert_resp(code=0, status=200, message="Prompts updated successfully")

    except Exception as e:
        logger.exception(f"Failed to update prompts: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to update prompts: {str(e)}")


@router.post("/api/settings/prompts/import")
async def import_prompts(file: UploadFile = File(...), _auth: str = auth_dependency):
    """Import prompts from YAML file"""
    try:
        prompt_mgr = GlobalConfig.get_instance().get_prompt_manager()
        if not prompt_mgr:
            return convert_resp(code=500, status=500, message="Prompt manager not initialized")

        # Read file content
        content = await file.read()
        yaml_content = content.decode("utf-8")

        if not prompt_mgr.import_prompts(yaml_content):
            return convert_resp(code=400, status=400, message="Failed to import prompts")

        logger.info("Prompts imported successfully")
        return convert_resp(code=0, status=200, message="Prompts imported successfully")

    except Exception as e:
        logger.exception(f"Failed to import prompts: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to import prompts: {str(e)}")


@router.get("/api/settings/prompts/export")
async def export_prompts(_auth: str = auth_dependency):
    """Export prompts as YAML file"""
    try:
        prompt_mgr = GlobalConfig.get_instance().get_prompt_manager()
        if not prompt_mgr:
            return convert_resp(code=500, status=500, message="Prompt manager not initialized")

        yaml_content = prompt_mgr.export_prompts()
        if not yaml_content:
            return convert_resp(code=500, status=500, message="Failed to export prompts")

        # Return as downloadable file
        language = GlobalConfig.get_instance().get_language()
        filename = f"prompts_{language}.yaml"

        return StreamingResponse(
            io.BytesIO(yaml_content.encode("utf-8")),
            media_type="application/x-yaml",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        logger.exception(f"Failed to export prompts: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to export prompts: {str(e)}")


@router.get("/api/settings/prompts/language")
async def get_prompt_language(_auth: str = auth_dependency):
    """Get current prompt language"""
    try:
        language = GlobalConfig.get_instance().get_language()
        return convert_resp(data={"language": language})
    except Exception as e:
        logger.exception(f"Failed to get prompt language: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to get language: {str(e)}")


class LanguageChangeRequest(BaseModel):
    """Language change request"""

    language: str = Field(..., pattern="^(zh|en)$")


@router.post("/api/settings/prompts/language")
async def change_prompt_language(request: LanguageChangeRequest, _auth: str = auth_dependency):
    """Change prompt language"""
    try:
        # Update language setting and reload prompts
        success = GlobalConfig.get_instance().set_language(request.language)

        if not success:
            return convert_resp(code=500, status=500, message="Failed to change language")

        logger.info(f"Prompt language changed to: {request.language}")
        return convert_resp(message=f"Language changed to {request.language}")
    except Exception as e:
        logger.exception(f"Failed to change prompt language: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to change language: {str(e)}")


# ==================== Reset Settings ====================


@router.post("/api/settings/reset")
async def reset_settings(_auth: str = auth_dependency):
    """Reset all user settings to defaults"""
    with _config_lock:
        try:
            config_mgr = GlobalConfig.get_instance().get_config_manager()
            prompt_mgr = GlobalConfig.get_instance().get_prompt_manager()

            success = True

            # Reset user settings
            if config_mgr:
                if not config_mgr.reset_user_settings():
                    success = False
                    logger.error("Failed to reset user settings")

            # Reset user prompts
            if prompt_mgr:
                if not prompt_mgr.reset_user_prompts():
                    success = False
                    logger.error("Failed to reset user prompts")

            if not success:
                return convert_resp(code=500, status=500, message="Failed to reset some settings")

            logger.info("All settings reset successfully")
            return convert_resp(code=0, status=200, message="Settings reset successfully")

        except Exception as e:
            logger.exception(f"Failed to reset settings: {e}")
            return convert_resp(code=500, status=500, message=f"Failed to reset settings: {str(e)}")
