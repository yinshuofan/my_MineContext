#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Settings API routes — model settings, general settings, prompts"""

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


class LLMModelConfig(BaseModel):
    """Flat model config for a single LLM/VLM/Embedding client"""

    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    max_concurrent: int | None = None


class EmbeddingModelConfig(LLMModelConfig):
    """Embedding model config with extra output_dim"""

    output_dim: int | None = None


class AllModelSettingsVO(BaseModel):
    """All 3 model configs returned by GET"""

    llm: LLMModelConfig | None = None
    vlm_model: LLMModelConfig | None = None
    embedding_model: EmbeddingModelConfig | None = None


class AllModelSettingsUpdateRequest(BaseModel):
    """Partial update — only non-None sections are saved"""

    llm: LLMModelConfig | None = None
    vlm_model: LLMModelConfig | None = None
    embedding_model: EmbeddingModelConfig | None = None


class ValidateModelRequest(BaseModel):
    """Validate one or more model sections"""

    llm: LLMModelConfig | None = None
    vlm_model: LLMModelConfig | None = None
    embedding_model: EmbeddingModelConfig | None = None


# ==================== Helper Functions ====================


def _build_llm_config(
    base_url: str, api_key: str, model: str, provider: str, llm_type: LLMType, **kwargs
) -> dict:
    """Build LLM config dict for LLMClient instantiation"""
    config = {"base_url": base_url, "api_key": api_key, "model": model, "provider": provider}
    if "timeout" in kwargs:
        config["timeout"] = kwargs["timeout"]
    if llm_type == LLMType.EMBEDDING:
        config["output_dim"] = kwargs.get("output_dim", 2048)
    return config


def _flatten_llm_config(config: dict) -> LLMModelConfig:
    """Flatten nested llm config {provider, config: {api_key, ...}} → flat LLMModelConfig"""
    nested = config.get("config", {})
    return LLMModelConfig(
        provider=config.get("provider", ""),
        model=nested.get("model", ""),
        base_url=nested.get("base_url", ""),
        api_key=nested.get("api_key", ""),
        max_concurrent=nested.get("max_concurrent"),
    )


def _nest_llm_config(flat: LLMModelConfig) -> dict:
    """Re-nest flat LLMModelConfig → {provider, config: {api_key, base_url, model, ...}}"""
    inner = {
        "api_key": flat.api_key,
        "base_url": flat.base_url,
        "model": flat.model,
    }
    if flat.max_concurrent is not None:
        inner["max_concurrent"] = flat.max_concurrent
    return {"provider": flat.provider, "config": inner}


def _flat_to_dict(cfg: LLMModelConfig) -> dict:
    """Convert flat LLMModelConfig → flat dict for vlm_model/embedding_model storage"""
    d = {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
    }
    if cfg.max_concurrent is not None:
        d["max_concurrent"] = cfg.max_concurrent
    if isinstance(cfg, EmbeddingModelConfig) and cfg.output_dim is not None:
        d["output_dim"] = cfg.output_dim
    return d


# ==================== Model Settings API ====================


@router.get("/api/model_settings/get")
async def get_model_settings(_auth: str = auth_dependency):
    """Get all 3 model configurations (llm, vlm, embedding)"""
    try:
        config = GlobalConfig.get_instance().get_config()
        if not config:
            return convert_resp(code=500, status=500, message="Configuration not initialized")

        llm_cfg = config.get("llm", {})
        vlm_cfg = config.get("vlm_model", {})
        emb_cfg = config.get("embedding_model", {})

        settings = AllModelSettingsVO(
            llm=_flatten_llm_config(llm_cfg),
            vlm_model=LLMModelConfig(
                provider=vlm_cfg.get("provider", ""),
                model=vlm_cfg.get("model", ""),
                base_url=vlm_cfg.get("base_url", ""),
                api_key=vlm_cfg.get("api_key", ""),
                max_concurrent=vlm_cfg.get("max_concurrent"),
            ),
            embedding_model=EmbeddingModelConfig(
                provider=emb_cfg.get("provider", ""),
                model=emb_cfg.get("model", ""),
                base_url=emb_cfg.get("base_url", ""),
                api_key=emb_cfg.get("api_key", ""),
                max_concurrent=emb_cfg.get("max_concurrent"),
                output_dim=emb_cfg.get("output_dim"),
            ),
        )

        return convert_resp(data=settings.model_dump())

    except Exception as e:
        logger.exception(f"Failed to get model settings: {e}")
        return convert_resp(code=500, status=500, message=f"Failed to get model settings: {str(e)}")


@router.post("/api/model_settings/update")
async def update_model_settings(
    request: AllModelSettingsUpdateRequest, _auth: str = auth_dependency
):
    """Update model configuration and reinitialize LLM clients"""
    with _config_lock:
        try:
            config_mgr = GlobalConfig.get_instance().get_config_manager()
            if not config_mgr:
                return convert_resp(code=500, status=500, message="Config manager not initialized")

            new_settings: dict = {}

            # --- LLM text model ---
            # No global singleton client for LLM text, so no reinitialize needed.
            # Config is read fresh from GlobalConfig at each usage point.
            if request.llm is not None:
                cfg = request.llm
                if cfg.api_key and cfg.model:
                    llm_config = _build_llm_config(
                        cfg.base_url,
                        cfg.api_key,
                        cfg.model,
                        cfg.provider,
                        LLMType.CHAT,
                        timeout=15,
                    )
                    llm_valid, llm_msg = LLMClient(
                        llm_type=LLMType.CHAT, config=llm_config
                    ).validate()
                    if not llm_valid:
                        return convert_resp(
                            code=400, status=400, message=f"LLM validation failed: {llm_msg}"
                        )
                new_settings["llm"] = _nest_llm_config(request.llm)

            # --- VLM ---
            if request.vlm_model is not None:
                cfg = request.vlm_model
                vlm_config = _build_llm_config(
                    cfg.base_url, cfg.api_key, cfg.model, cfg.provider, LLMType.CHAT, timeout=15
                )
                vlm_valid, vlm_msg = LLMClient(
                    llm_type=LLMType.CHAT, config=vlm_config
                ).validate()
                if not vlm_valid:
                    return convert_resp(
                        code=400, status=400, message=f"VLM validation failed: {vlm_msg}"
                    )
                new_settings["vlm_model"] = _flat_to_dict(cfg)

            # --- Embedding ---
            if request.embedding_model is not None:
                cfg = request.embedding_model
                emb_config = _build_llm_config(
                    cfg.base_url,
                    cfg.api_key,
                    cfg.model,
                    cfg.provider,
                    LLMType.EMBEDDING,
                    timeout=15,
                    output_dim=cfg.output_dim or 2048,
                )
                emb_valid, emb_msg = LLMClient(
                    llm_type=LLMType.EMBEDDING, config=emb_config
                ).validate()
                if not emb_valid:
                    return convert_resp(
                        code=400, status=400, message=f"Embedding validation failed: {emb_msg}"
                    )
                new_settings["embedding_model"] = _flat_to_dict(cfg)

            if not new_settings:
                return convert_resp(code=400, status=400, message="No model settings provided")

            if not config_mgr.save_user_settings(new_settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

            config_mgr.load_config(config_mgr.get_config_path())

            # Reinitialize clients that were updated
            if "vlm_model" in new_settings:
                if not GlobalVLMClient.get_instance().reinitialize():
                    return convert_resp(
                        code=500, status=500, message="Failed to reinitialize VLM client"
                    )
            if "embedding_model" in new_settings:
                if not GlobalEmbeddingClient.get_instance().reinitialize():
                    return convert_resp(
                        code=500, status=500, message="Failed to reinitialize embedding client"
                    )

            logger.info("Model settings updated successfully")
            return convert_resp(
                data={"success": True, "message": "Model settings updated successfully"}
            )

        except Exception as e:
            logger.exception(f"Failed to update model settings: {e}")
            return convert_resp(code=500, status=500, message="Failed to update model settings")


@router.post("/api/model_settings/validate")
async def validate_llm_config(request: ValidateModelRequest, _auth: str = auth_dependency):
    """Validate LLM configuration without saving"""
    try:
        errors = []
        validated_count = 0

        if request.llm is not None:
            cfg = request.llm
            if cfg.api_key and cfg.model:
                validated_count += 1
                llm_config = _build_llm_config(
                    cfg.base_url, cfg.api_key, cfg.model, cfg.provider, LLMType.CHAT, timeout=15
                )
                valid, msg = LLMClient(llm_type=LLMType.CHAT, config=llm_config).validate()
                if not valid:
                    errors.append(f"LLM: {msg}")

        if request.vlm_model is not None:
            cfg = request.vlm_model
            if cfg.api_key and cfg.model:
                validated_count += 1
                vlm_config = _build_llm_config(
                    cfg.base_url, cfg.api_key, cfg.model, cfg.provider, LLMType.CHAT, timeout=15
                )
                valid, msg = LLMClient(llm_type=LLMType.CHAT, config=vlm_config).validate()
                if not valid:
                    errors.append(f"VLM: {msg}")

        if request.embedding_model is not None:
            cfg = request.embedding_model
            if cfg.api_key and cfg.model:
                validated_count += 1
                emb_config = _build_llm_config(
                    cfg.base_url,
                    cfg.api_key,
                    cfg.model,
                    cfg.provider,
                    LLMType.EMBEDDING,
                    timeout=15,
                    output_dim=cfg.output_dim or 2048,
                )
                valid, msg = LLMClient(llm_type=LLMType.EMBEDDING, config=emb_config).validate()
                if not valid:
                    errors.append(f"Embedding: {msg}")

        if errors:
            return convert_resp(code=400, status=400, message="; ".join(errors))

        if validated_count == 0:
            return convert_resp(
                code=400, status=400, message="No credentials provided — fill in API key and model"
            )

        return convert_resp(code=0, status=200, message="Connection test passed")

    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        return convert_resp(code=500, status=500, message=f"Validation failed: {str(e)}")


# ==================== General Settings ====================


class GeneralSettingsRequest(BaseModel):
    """General system settings request — all sections optional"""

    capture: dict | None = None
    processing: dict | None = None
    logging: dict | None = None
    document_processing: dict | None = None
    scheduler: dict | None = None
    memory_cache: dict | None = None
    tools: dict | None = None


_GENERAL_SETTINGS_KEYS = [
    "capture",
    "processing",
    "logging",
    "document_processing",
    "scheduler",
    "memory_cache",
    "tools",
]


@router.get("/api/settings/general")
async def get_general_settings(_auth: str = auth_dependency):
    """Get general system settings"""
    try:
        config = GlobalConfig.get_instance().get_config()
        if not config:
            return convert_resp(code=500, status=500, message="Configuration not initialized")

        settings = {key: config.get(key, {}) for key in _GENERAL_SETTINGS_KEYS}
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

            settings = {}
            req_dict = request.model_dump(exclude_none=True)
            for key in _GENERAL_SETTINGS_KEYS:
                if key in req_dict:
                    settings[key] = req_dict[key]

            if not settings:
                return convert_resp(code=400, status=400, message="No settings provided")

            if not config_mgr.save_user_settings(settings):
                return convert_resp(code=500, status=500, message="Failed to save settings")

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

            if config_mgr:
                if not config_mgr.reset_user_settings():
                    success = False
                    logger.error("Failed to reset user settings")

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
            return convert_resp(
                code=500, status=500, message=f"Failed to reset settings: {str(e)}"
            )
