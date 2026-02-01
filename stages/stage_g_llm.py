"""
Stage G: LLM Prompt Synthesis

Uses an LLM to:
- Unify style, motion, and subject into a clean generation prompt
- Avoid contradictions
- Produce main prompt and negative prompt
- Keep output focused on 2-4 second clips

Supports:
- MiniMax M2.1 (recommended - best coding/instruction following)
- OpenAI (gpt-4o)
- Anthropic (claude-3-5-sonnet)
"""
import json
import logging
import os
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import LLMConfig, ContextTemplate
from schemas import SceneReport, LLMPromptOutput

logger = logging.getLogger(__name__)


# System prompt template
# Note: Double curly braces {{ }} are escaped for Python .format()
SYSTEM_PROMPT = """You are an expert at creating video generation prompts from structured scene analysis.

Your task: Convert a scene analysis report into a cinematic prompt suitable for video regeneration.

CRITICAL RULES:
1. Only include details that appear in the scene report. If uncertain, omit or hedge.
2. Never invent animals, people, or objects not supported by evidence.
3. Maintain documentary realism - this is underwater ROV footage, not CGI.
4. Prioritize camera motion accuracy - the motion descriptors are reliable.
5. Keep the prompt concise but descriptive (2-4 sentences for 2-4 sec clip).
6. The negative prompt should exclude things that would break realism.

OUTPUT FORMAT:
You must respond with valid JSON in this exact format:
{{
    "final_prompt": "Your cinematic prompt here...",
    "negative_prompt": "Things to avoid...",
    "confidence_notes": ["Note about any uncertainties..."],
    "used_evidence": ["List of evidence you relied on..."],
    "suggested_duration_sec": 3.0,
    "style_tags": ["documentary", "underwater", "etc"]
}}

CONTEXT ABOUT THE FOOTAGE:
{context}

Remember: Faithfulness to the scene report is paramount. A conservative, accurate prompt is better than an imaginative but inaccurate one."""


USER_PROMPT_TEMPLATE = """Please generate a video generation prompt from this scene analysis:

{scene_report}

Focus on:
1. Accurately describing the camera motion: {motion_summary}
2. The main subject: {subject_summary}
3. The underwater environment: {environment_summary}

Generate a prompt that would faithfully recreate this footage."""


class MiniMaxClient:
    """
    Client for MiniMax API (M2.1 model).
    
    MiniMax M2.1 features:
    - Exceptional multi-language programming capabilities
    - Enhanced composite instruction constraints
    - More concise and efficient responses
    - Outstanding Agent/Tool scaffolding generalization
    
    Reference: https://www.minimax.io/news/minimax-m21
    """
    
    def __init__(self, model: str = "MiniMax-M2.1"):
        self.api_key = os.getenv("MINIMAX_API_KEY", "")
        self.group_id = os.getenv("MINIMAX_GROUP_ID", "")
        self.base_url = "https://api.minimax.io/v1/chat/completions"
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY not set.\n"
                "Get your API key from: https://platform.minimax.io\n"
                "Set it with: $env:MINIMAX_API_KEY='your-key'"
            )
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """
        Make a chat completion request to MiniMax API.
        
        Args:
            system_prompt: System context message
            user_prompt: User message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build URL with group_id if provided
        url = self.base_url
        if self.group_id:
            url = f"{url}?GroupId={self.group_id}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # MiniMax recommended parameters for best performance
            "top_p": 0.95,
            "top_k": 40
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            result = response.json()
            
            # Check for API errors
            if "error" in result:
                logger.error(f"MiniMax API error: {result['error']}")
                return None
            
            # Check base_resp for MiniMax-specific errors
            if "base_resp" in result:
                base_resp = result["base_resp"]
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    logger.error(
                        f"MiniMax error ({status_code}): {base_resp.get('status_msg', 'Unknown')}"
                    )
                    return None
            
            # Extract response content
            if "choices" not in result or len(result["choices"]) == 0:
                logger.error(f"Unexpected MiniMax response: {result}")
                return None
            
            content = result["choices"][0]["message"]["content"]
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MiniMax API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"MiniMax API error: {e}")
            return None


class LLMPromptSynthesizer:
    """
    Stage G: Use LLM to synthesize final generation prompts.
    
    Supports MiniMax M2.1, OpenAI, and Anthropic APIs.
    MiniMax M2.1 is recommended for best instruction following.
    """
    
    def __init__(
        self, 
        config: Optional[LLMConfig] = None,
        context: Optional[ContextTemplate] = None
    ):
        self.config = config or LLMConfig()
        self.context = context or ContextTemplate()
        self._client = None
    
    def _get_context_string(self) -> str:
        """Format context template for system prompt."""
        return f"""- Environment: {self.context.domain}, {self.context.platform} camera
- Style: {self.context.camera_style}
- Priorities: {', '.join(self.context.priorities)}
- Forbidden: {', '.join(self.context.forbidden)}
- Target duration: {self.context.target_duration}"""
    
    def _init_minimax(self) -> bool:
        """Initialize MiniMax client."""
        try:
            self._client = MiniMaxClient(model=self.config.model)
            return True
        except ValueError as e:
            logger.warning(f"MiniMax initialization failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MiniMax: {e}")
            return False
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI()
            return True
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False
    
    def _init_anthropic(self) -> bool:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            return True
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            return False
    
    def _call_minimax(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Optional[str]:
        """Call MiniMax API."""
        return self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _call_openai(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Optional[str]:
        """Call OpenAI API."""
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None
    
    def _call_anthropic(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> Optional[str]:
        """Call Anthropic API."""
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None
    
    def _extract_summaries(self, report: SceneReport) -> Dict[str, str]:
        """Extract key summaries from scene report for the prompt."""
        # Motion summary
        if report.camera_motion:
            motion_parts = []
            for seg in report.camera_motion:
                label = " + ".join(seg.labels)
                motion_parts.append(f"{label} ({seg.t0:.1f}s-{seg.t1:.1f}s)")
            motion_summary = "; ".join(motion_parts)
        else:
            motion_summary = "No motion data available"
        
        # Subject summary
        subj = report.main_subject
        subject_summary = subj.hypothesis
        if subj.appearance:
            subject_summary += f" ({', '.join(subj.appearance[:3])})"
        
        # Environment summary
        env = report.environment
        env_parts = []
        if env.water_color != "unknown":
            env_parts.append(f"water: {env.water_color}")
        if env.visibility != "unknown":
            env_parts.append(f"visibility: {env.visibility}")
        if env.seafloor:
            env_parts.append(f"seafloor: {', '.join(env.seafloor[:2])}")
        environment_summary = "; ".join(env_parts) if env_parts else "underwater environment"
        
        return {
            "motion_summary": motion_summary,
            "subject_summary": subject_summary,
            "environment_summary": environment_summary
        }
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or extra text."""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        # Find JSON object boundaries
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text
    
    def synthesize(self, report: SceneReport) -> LLMPromptOutput:
        """
        Synthesize final prompt from scene report using LLM.
        
        Args:
            report: Complete scene report from Stage F
            
        Returns: LLMPromptOutput with final and negative prompts
        """
        # Try to initialize API client based on provider
        init_success = False
        
        if self.config.provider == "minimax":
            init_success = self._init_minimax()
        elif self.config.provider == "openai":
            init_success = self._init_openai()
        elif self.config.provider == "anthropic":
            init_success = self._init_anthropic()
        else:
            logger.warning(f"Unknown provider: {self.config.provider}, trying MiniMax")
            self.config.provider = "minimax"
            init_success = self._init_minimax()
        
        if not init_success:
            logger.warning("LLM initialization failed, using fallback synthesis")
            return self._fallback_synthesis(report)
        
        # Build prompts
        system_prompt = SYSTEM_PROMPT.format(context=self._get_context_string())
        
        summaries = self._extract_summaries(report)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            scene_report=json.dumps(report.to_llm_payload(), indent=2),
            **summaries
        )
        
        # Call LLM based on provider
        response = None
        if self.config.provider == "minimax":
            response = self._call_minimax(system_prompt, user_prompt)
        elif self.config.provider == "openai":
            response = self._call_openai(system_prompt, user_prompt)
        elif self.config.provider == "anthropic":
            response = self._call_anthropic(system_prompt, user_prompt)
        
        if response is None:
            logger.warning("LLM call failed, using fallback synthesis")
            return self._fallback_synthesis(report)
        
        # Parse response
        try:
            json_text = self._extract_json(response)
            data = json.loads(json_text)
            return LLMPromptOutput(
                final_prompt=data.get("final_prompt", ""),
                negative_prompt=data.get("negative_prompt", ""),
                confidence_notes=data.get("confidence_notes", []),
                used_evidence=data.get("used_evidence", []),
                suggested_duration_sec=data.get("suggested_duration_sec", 3.0),
                style_tags=data.get("style_tags", [])
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Try to extract prompt from raw response
            return LLMPromptOutput(
                final_prompt=response[:500],
                negative_prompt="",
                confidence_notes=["Failed to parse structured response"]
            )
    
    def _fallback_synthesis(self, report: SceneReport) -> LLMPromptOutput:
        """
        Generate prompt without LLM (rule-based fallback).
        
        Used when LLM is unavailable.
        """
        logger.info("Using fallback prompt synthesis (no LLM)")
        
        # Build prompt from report components
        parts = []
        
        # Camera motion
        if report.camera_motion:
            motion_descs = []
            for seg in report.camera_motion[:3]:  # First 3 segments
                label = " and ".join(seg.labels).replace("_", " ")
                motion_descs.append(label)
            parts.append(f"Camera motion: {', then '.join(motion_descs)}.")
        
        # Environment
        env = report.environment
        env_desc = f"{env.water_color} water with {env.visibility}"
        if env.particulate and env.particulate != "unknown":
            env_desc += f", {env.particulate}"
        parts.append(f"Environment: {env_desc}.")
        
        # Subject
        subj = report.main_subject
        if subj.hypothesis and "unknown" not in subj.hypothesis.lower():
            subj_desc = subj.hypothesis
            if subj.appearance:
                subj_desc += f" ({', '.join(subj.appearance[:2])})"
            parts.append(f"Subject: {subj_desc}.")
        
        # Style
        parts.append(f"Style: {report.context.camera_style}, documentary underwater footage.")
        
        final_prompt = " ".join(parts)
        
        # Build negative prompt
        negatives = ["blurry", "shaky", "artificial", "CGI"]
        if not report.negatives.people_detected:
            negatives.append("people")
            negatives.append("divers")
        if report.negatives.fish_detected is False:
            negatives.append("fish")
            negatives.append("marine life")
        
        negative_prompt = ", ".join(negatives)
        
        return LLMPromptOutput(
            final_prompt=final_prompt,
            negative_prompt=negative_prompt,
            confidence_notes=["Generated using rule-based fallback (no LLM)"],
            used_evidence=[
                f"motion_segments: {len(report.camera_motion)}",
                f"keyframes: {len(report.keyframes)}",
                f"subject_confidence: {report.main_subject.confidence:.0%}"
            ],
            suggested_duration_sec=min(report.duration_sec, 4.0),
            style_tags=["documentary", "underwater", "ROV", report.context.camera_style]
        )


def synthesize_prompt(
    report: SceneReport,
    config: Optional[LLMConfig] = None,
    context: Optional[ContextTemplate] = None
) -> LLMPromptOutput:
    """
    Convenience function to synthesize prompt from scene report.
    """
    synthesizer = LLMPromptSynthesizer(config, context)
    return synthesizer.synthesize(report)
