"""
VLM Verification Module
Sends evidence frames to a vision-capable LLM for dumping event confirmation.
Uses OpenRouter (OpenAI-compatible API) as the backend.
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
from dotenv import load_dotenv
from openai import OpenAI

# Load .env if present
load_dotenv()


@dataclass
class VerificationResult:
    confirmed: bool
    event_type: str
    severity: str  # LOW, MEDIUM, HIGH
    summary: str
    raw_response: str
    latency_ms: float
    model: str
    verified: bool  # True if VLM responded, False if fallback


PROMPT = """You are a surveillance video analyst verifying whether an unattended object is illegal dumping.

A computer vision system has flagged an object as potentially abandoned. Your job is to determine whether this object is consistent with illegal dumping or littering.

EVIDENCE: The image shows a candidate object (highlighted with a red bounding box labeled CANDIDATE) in its surrounding context.

DECISION CRITERIA — confirm ILLEGAL DUMPING if the object appears to be:
- Garbage bags, trash bags, or waste bags left in a public area
- Discarded household waste or commercial refuse
- Abandoned cardboard, packaging, or containers
- Discarded furniture, appliances, or large items
- Construction debris or materials dumped inappropriately
- Any object that appears to be discarded waste/trash left behind by a person who departed

REJECT (not dumping) if the object appears to be:
- Sports equipment (balls, rackets, etc.)
- Personal belongings temporarily placed (luggage, shopping bags being carried)
- An object being actively used or handled by a person
- A normal fixture of the environment (sign, post, bench, planter)
- An object that is clearly in motion or being transported
- Anything that does not visually resemble discarded waste

RESPOND WITH ONLY a JSON object (no markdown, no extra text):
{
    "confirmed": true/false,
    "event_type": "illegal_dumping" | "abandoned_object" | "normal_scene" | "unclear",
    "severity": "LOW" | "MEDIUM" | "HIGH",
    "summary": "Brief 1-sentence description of what you observe"
}"""


def encode_frame_to_base64(frame: cv2.Mat, quality: int = 80) -> str:
    """Encode an OpenCV frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def crop_candidate_evidence(frame: cv2.Mat, bbox: tuple) -> cv2.Mat:
    """Crop frame around candidate bbox with context padding and highlight the candidate."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    pad_x = max(int(bw * 0.5), 30)
    pad_y = max(int(bh * 0.5), 30)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2].copy()

    rx1, ry1 = x1 - cx1, y1 - cy1
    rx2, ry2 = x2 - cx1, y2 - cy1
    cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
    cv2.putText(crop, "CANDIDATE", (rx1, ry1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return crop


def verify_dumping_event(
    frame: cv2.Mat,
    track_id: int,
    class_name: str,
    centroid: tuple,
    bbox: Optional[tuple] = None,
    confidence: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    timeout: int = 30,
) -> VerificationResult:
    """
    Send an evidence frame to a VLM for dumping event verification.

    Returns VerificationResult with confirmed/event_type/severity/summary.
    On any failure, returns a fallback result with verified=False.
    """
    start = time.time()

    # Get API credentials
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not key:
        latency = (time.time() - start) * 1000
        return VerificationResult(
            confirmed=False,
            event_type="unverified",
            severity="LOW",
            summary="VLM unavailable: no API key configured",
            raw_response="",
            latency_ms=latency,
            model=model,
            verified=False,
        )

    # Encode frame — crop to candidate if bbox available
    if bbox is not None:
        evidence_frame = crop_candidate_evidence(frame, bbox)
    else:
        evidence_frame = frame
    img_b64 = encode_frame_to_base64(evidence_frame)

    # Build message
    user_content = [
        {
            "type": "text",
            "text": PROMPT,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "low",
            },
        },
    ]

    try:
        client = OpenAI(api_key=key, base_url=url, timeout=timeout)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a surveillance video analyst. Respond only with valid JSON."},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1000,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        latency = (time.time() - start) * 1000

        # Parse JSON — handle markdown code blocks if present
        cleaned = raw
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        data = json.loads(cleaned)

        return VerificationResult(
            confirmed=bool(data.get("confirmed", False)),
            event_type=str(data.get("event_type", "unknown")),
            severity=str(data.get("severity", "LOW")),
            summary=str(data.get("summary", "No summary")),
            raw_response=raw,
            latency_ms=latency,
            model=model,
            verified=True,
        )

    except json.JSONDecodeError as e:
        latency = (time.time() - start) * 1000
        return VerificationResult(
            confirmed=False,
            event_type="parse_error",
            severity="LOW",
            summary=f"VLM returned invalid JSON: {e}",
            raw_response=raw if "raw" in dir() else "",
            latency_ms=latency,
            model=model,
            verified=False,
        )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return VerificationResult(
            confirmed=False,
            event_type="error",
            severity="LOW",
            summary=f"VLM request failed: {type(e).__name__}",
            raw_response="",
            latency_ms=latency,
            model=model,
            verified=False,
        )
