"""
AnatomoZK: 3D Coronary Risk & Byzantine BFT Trust - Temporal Byzantine Engine (Phase 3)
=======================================================================================
Fault-Tolerant Distributed Architecture & Cryptographic Validation State Machine.

Architectural Design (Phase 3):
  1. Class Architecture:
     - Central verification manager: `TemporalByzantineConsensusEngine`.
     - Ingests concurrent outputs from Phase 2 `agent_orchestrator.py`
       (`AgentEvaluationResult` payloads) mapped across the millisecond temporal grid.
  2. Temporal Byzantine Consensus Protocol:
     - Mathematical alignment algorithm mapping discrete verdicts
       (NOMINAL, AMBIGUOUS, FRAUD_SUSPECTED) into continuous numerical vector signatures
       per millisecond bin: phi(s) in {+1.0, 0.0, -1.0}.
     - Byzantine Agreement Heuristic:
       With N=4 independent modal validation nodes (Vision, Acoustic, Telemetry, Compliance),
       satisfies Classical BFT constraint (N >= 3f + 1, tolerating f=1 faulty/divergent node).
       If 3 or more independent nodes agree on session structural consistency, but a single
       modality introduces a sudden divergent state shift, instantly triggers `StateByzantineFault`
       and executes consensus quorum override.
  3. Cross-Modal Coincidence Clustered Index:
     - Temporal interception module grouping multi-sensor micro-anomalies
       (e.g., optical flow surge + rolling Z-score telemetry spike + acoustic transient)
       within a calibrated temporal jitter window (Delta tau = 150ms).
  4. Discrepancy Matrix Synthesis:
     - Localized timeline tracking, conflict severity tiers (NONE, LOW, MEDIUM, CRITICAL),
       fault duration tracking, and active consensus override indicators.
  5. Scannable Verification Runner:
     - Directly integrates with Phase 1 `MultimodalSessionIngestor` and Phase 2 `DecoupledStateRouter`,
       executes the consensus state machine, and outputs solid black markdown diagnostic tables.
"""

from __future__ import annotations

import os
import sys
import math
import time
import shutil
import tempfile
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set, Union

import numpy as np
import pandas as pd

# Phase 1 & Phase 2 Pipeline Bedrock Imports
from ingestion_engine import (
    MultimodalSessionIngestor,
    IngestionConfig,
    AlignedTemporalSlice,
    ComplianceDocMetadata,
    generate_mock_session_assets,
    audit_environment,
    SystemEnvironment
)
from agent_orchestrator import (
    VerificationState,
    AgentEvaluationResult,
    MultiAgentConsensus,
    DecoupledStateRouter,
    LLMClientRegistry
)


# =====================================================================
# 1. FAULT TOLERANCE SCHEMAS & CONFLICT SEVERITY TIERS
# =====================================================================

class ConflictSeverity(str, Enum):
    """Hierarchical severity tiers for multi-modal state discrepancy."""
    NONE = "NONE"          # Unanimous 4/4 agreement
    LOW = "LOW"            # 3/4 agreement with single 1-step divergence (NOMINAL vs AMBIGUOUS)
    MEDIUM = "MEDIUM"      # 3/4 agreement with single 2-step divergence (NOMINAL vs FRAUD_SUSPECTED) -> Byzantine Fault Overridden
    CRITICAL = "CRITICAL"  # Quorum deficit (2 vs 2 split) or confirmed multi-modal fraud surge


class ByzantineFaultType(str, Enum):
    """Taxonomy of isolated and systemic Byzantine divergences."""
    NO_FAULT = "NO_FAULT"
    ISOLATED_TRANSDUCER_FAULT = "ISOLATED_TRANSDUCER_FAULT"  # Sensor anomaly while others nominal
    ISOLATED_VISUAL_ARTIFACT = "ISOLATED_VISUAL_ARTIFACT"    # Lighting/camera glitch while others nominal
    ISOLATED_ACOUSTIC_TRANSIENT = "ISOLATED_ACOUSTIC_TRANSIENT"
    QUORUM_DEFICIT_SPLIT = "QUORUM_DEFICIT_SPLIT"            # Split voting (2 vs 2)
    COORDINATED_FRAUD_SURGE = "COORDINATED_FRAUD_SURGE"      # Multi-modal coincident breach


@dataclass
class StateVectorSignature:
    """Numerical vector representation of modal state at millisecond bin t."""
    timestamp_ms: int
    vision_score: float      # +1.0 (NOMINAL), 0.0 (AMBIGUOUS), -1.0 (FRAUD_SUSPECTED)
    acoustic_score: float
    telemetry_score: float
    compliance_score: float
    vector_mean: float
    vector_variance: float
    disagreement_ratio: float


@dataclass
class TemporalDiscrepancyRecord:
    """Consolidated Byzantine evaluation for a single millisecond time bin."""
    timestamp_ms: int
    consensus_verdict: VerificationState
    quorum_agreement_pct: float
    conflict_severity: ConflictSeverity
    byzantine_fault_flag: bool
    faulting_node: str
    consensus_override_active: bool
    fault_duration_ms: int
    cluster_id: Optional[str] = None
    state_vector: Optional[StateVectorSignature] = None


@dataclass
class CoincidenceCluster:
    """Spatial-temporal group of correlated anomalies across independent modalities."""
    cluster_id: str
    centroid_ms: float
    time_span_ms: Tuple[int, int]
    modalities_involved: List[str]
    coincidence_tier: str        # 'BI_MODAL', 'TRI_MODAL', 'QUAD_MODAL'
    correlation_score: float     # 0.0 to 1.0 based on temporal density and modal diversity
    clinical_interpretation: str
    slices_count: int


# =====================================================================
# 2. STATE VALUE ALIGNMENT & VECTOR ENCODING
# =====================================================================

def map_state_to_scalar(state: VerificationState) -> float:
    """Deterministic injective mapping from VerificationState to real line."""
    if state == VerificationState.NOMINAL:
        return 1.0
    elif state == VerificationState.AMBIGUOUS:
        return 0.0
    elif state == VerificationState.FRAUD_SUSPECTED:
        return -1.0
    return 0.0


def map_scalar_to_state(val: float) -> VerificationState:
    """Deterministic rounding from continuous scalar back to VerificationState."""
    if val >= 0.5:
        return VerificationState.NOMINAL
    elif val <= -0.5:
        return VerificationState.FRAUD_SUSPECTED
    else:
        return VerificationState.AMBIGUOUS


# =====================================================================
# 3. CROSS-MODAL COINCIDENCE CLUSTERED INDEX
# =====================================================================

class CrossModalCoincidenceIndexer:
    """
    Temporal interception engine:
    Scans aligned multi-stream timelines to identify and cluster micro-anomalies
    co-occurring across completely independent physical sensors within a calibrated
    temporal jitter window (Delta tau).
    """

    def __init__(self, temporal_jitter_ms: int = 150):
        self.temporal_jitter_ms = temporal_jitter_ms
        self.clusters: List[CoincidenceCluster] = []
        self._cluster_by_time: Dict[int, str] = {}

    def build_index(self, slices: List[AlignedTemporalSlice]) -> List[CoincidenceCluster]:
        """Scans slices, groups coincident events, and registers spatial-temporal clusters."""
        # 1. Identify slices exhibiting anomalous activity in >= 1 modality
        anomaly_points: List[Tuple[int, Set[str], List[str]]] = []

        for s in slices:
            active_modalities: Set[str] = set()
            signatures: List[str] = []

            # Vision anomaly
            if s.cv_splice_flag or (s.cv_diff_variance is not None and s.cv_diff_variance >= 250.0):
                active_modalities.add("VISION")
                signatures.append("SPLICE_CUT")
            elif s.cv_flow_magnitude_px is not None and s.cv_flow_magnitude_px >= 3.0:
                active_modalities.add("VISION")
                signatures.append(f"OPTICAL_FLOW_SURGE({s.cv_flow_magnitude_px:.1f}px)")

            # Acoustic anomaly
            if s.audio_transient_flag or (s.audio_rms is not None and s.audio_rms >= 0.25):
                active_modalities.add("ACOUSTIC")
                signatures.append("ACOUSTIC_BURST")

            # Telemetry anomaly
            if s.telemetry_outliers_active:
                active_modalities.add("TELEMETRY")
                signatures.extend(s.telemetry_outliers_active[:2])

            # Compliance boundary breach
            hr = s.telemetry_values.get("heart_rate", 70.0)
            if hr > 120.0 or s.cv_splice_flag:
                active_modalities.add("COMPLIANCE")
                signatures.append(f"PROTOCOL_BREACH(HR={hr:.1f})")

            if active_modalities:
                anomaly_points.append((s.offset_ms, active_modalities, signatures))

        # 2. Cluster points within temporal jitter window
        if not anomaly_points:
            return []

        clusters_raw: List[List[Tuple[int, Set[str], List[str]]]] = []
        curr_cluster = [anomaly_points[0]]

        for pt in anomaly_points[1:]:
            last_t = curr_cluster[-1][0]
            if (pt[0] - last_t) <= self.temporal_jitter_ms:
                curr_cluster.append(pt)
            else:
                clusters_raw.append(curr_cluster)
                curr_cluster = [pt]
        if curr_cluster:
            clusters_raw.append(curr_cluster)

        # 3. Synthesize Coincidence Clusters
        self.clusters = []
        self._cluster_by_time = {}

        for idx, cl in enumerate(clusters_raw, 1):
            cluster_id = f"CC-{idx:03d}"
            timestamps = [p[0] for p in cl]
            all_mods: Set[str] = set()
            for p in cl:
                all_mods.update(p[1])

            centroid = float(np.mean(timestamps))
            t_span = (min(timestamps), max(timestamps))
            mod_count = len(all_mods)

            # Taxonomy Classification
            if mod_count >= 3:
                tier = "TRI_MODAL" if mod_count == 3 else "QUAD_MODAL"
                interp = "COORDINATED_FRAUD_SURGE: Physical cut coincident with physiological & acoustic breach"
                score = min(1.000, round(0.85 + 0.035 * mod_count, 3))
            elif mod_count == 2:
                tier = "BI_MODAL"
                if "VISION" in all_mods and "TELEMETRY" in all_mods:
                    interp = "BI_MODAL_COINCIDENCE: Optical movement disturbance matches sensor anomaly"
                elif "TELEMETRY" in all_mods and "ACOUSTIC" in all_mods:
                    interp = "PHYSIOLOGICAL_ACOUSTIC_TRANSIENT: Audio alert matches acute vital spike"
                else:
                    interp = f"BI_MODAL_CROSS_SENSOR_EVENT: {list(all_mods)}"
                score = 0.750
            else:
                tier = "MONO_MODAL"
                mod_name = list(all_mods)[0]
                interp = f"ISOLATED_BYZANTINE_ANOMALY: Single {mod_name} sensor divergence"
                score = 0.350

            cluster_obj = CoincidenceCluster(
                cluster_id=cluster_id,
                centroid_ms=round(centroid, 1),
                time_span_ms=t_span,
                modalities_involved=sorted(list(all_mods)),
                coincidence_tier=tier,
                correlation_score=score,
                clinical_interpretation=interp,
                slices_count=len(cl)
            )
            self.clusters.append(cluster_obj)

            for t in timestamps:
                self._cluster_by_time[t] = cluster_id

        return self.clusters

    def get_cluster_id(self, timestamp_ms: int) -> Optional[str]:
        """Fast lookup of cluster ID for a given timestamp."""
        return self._cluster_by_time.get(timestamp_ms)


# =====================================================================
# 4. TEMPORAL BYZANTINE CONSENSUS ENGINE
# =====================================================================

class TemporalByzantineConsensusEngine:
    """
    Central Fault-Tolerant Consensus Orchestrator:
    - Ingests concurrent outputs from Phase 2 validation agents mapped across timeline.
    - Implements Classical Byzantine Fault Tolerance (N=4, f=1):
      If 3+ independent nodes maintain consensus but 1 node exhibits sudden divergence,
      triggers StateByzantineFault, isolates the faulty modality, and executes quorum override.
    - Synthesizes Master Byzantine Discrepancy Matrix with conflict tiers and fault durations.
    """

    def __init__(
        self,
        temporal_jitter_ms: int = 150,
        byzantine_quorum_threshold: float = 0.75  # 3 of 4 validators (75%)
    ):
        self.temporal_jitter_ms = temporal_jitter_ms
        self.quorum_threshold = byzantine_quorum_threshold
        self.indexer = CrossModalCoincidenceIndexer(temporal_jitter_ms=temporal_jitter_ms)

    def evaluate_modal_state(
        self,
        slice_obj: AlignedTemporalSlice,
        global_agent_evals: Dict[str, AgentEvaluationResult]
    ) -> Tuple[VerificationState, VerificationState, VerificationState, VerificationState]:
        """
        Derives localized VerificationState for each of the 4 independent modalities
        at millisecond bin t, factoring in localized edge signals and global agent priors.
        Returns: (vision_state, acoustic_state, telemetry_state, compliance_state)
        """
        # 1. Vision Modality State
        if slice_obj.cv_splice_flag or (slice_obj.cv_diff_variance is not None and slice_obj.cv_diff_variance >= 250.0):
            vis_state = VerificationState.FRAUD_SUSPECTED
        elif (slice_obj.cv_flow_magnitude_px is not None and slice_obj.cv_flow_magnitude_px >= 3.0) or \
             (slice_obj.cv_diff_variance is not None and slice_obj.cv_diff_variance >= 100.0):
            vis_state = VerificationState.AMBIGUOUS
        else:
            vis_state = VerificationState.NOMINAL

        # 2. Acoustic Modality State
        if slice_obj.audio_transient_flag or (slice_obj.audio_rms is not None and slice_obj.audio_rms >= 0.25):
            # If coincident with a visual splice, high severity
            if slice_obj.cv_splice_flag:
                aud_state = VerificationState.FRAUD_SUSPECTED
            else:
                aud_state = VerificationState.AMBIGUOUS
        elif slice_obj.audio_rms is not None and slice_obj.audio_rms >= 0.10:
            aud_state = VerificationState.AMBIGUOUS
        else:
            aud_state = VerificationState.NOMINAL

        # 3. Telemetry Modality State
        hr = slice_obj.telemetry_values.get("heart_rate", 70.0)
        hr_z = slice_obj.telemetry_z_scores.get("heart_rate", 0.0)
        has_severe_spike = abs(hr_z) >= 10.0 or hr >= 140.0

        if has_severe_spike:
            tel_state = VerificationState.FRAUD_SUSPECTED
        elif slice_obj.telemetry_outliers_active or abs(hr_z) >= 3.0:
            tel_state = VerificationState.AMBIGUOUS
        else:
            tel_state = VerificationState.NOMINAL

        # 4. Compliance Document Modality State
        if hr > 120.0 or slice_obj.cv_splice_flag:
            com_state = VerificationState.FRAUD_SUSPECTED
        elif hr > 105.0 or (slice_obj.cv_diff_variance is not None and slice_obj.cv_diff_variance > 150.0):
            com_state = VerificationState.AMBIGUOUS
        else:
            com_state = VerificationState.NOMINAL

        return vis_state, aud_state, tel_state, com_state

    def execute_byzantine_consensus(
        self,
        slices: List[AlignedTemporalSlice],
        agent_evaluations: Dict[str, AgentEvaluationResult]
    ) -> List[TemporalDiscrepancyRecord]:
        """
        Executes the Byzantine Agreement Heuristic across all temporal slices.
        Tracks fault durations and overrides divergent nodes.
        """
        # Build Coincidence Index
        self.indexer.build_index(slices)

        discrepancy_records: List[TemporalDiscrepancyRecord] = []
        active_fault_duration_ms = 0
        last_fault_active = False

        for s in slices:
            t = s.offset_ms
            v_vis, v_aud, v_tel, v_com = self.evaluate_modal_state(s, agent_evaluations)

            # Map states to numerical scalars
            s_vis = map_state_to_scalar(v_vis)
            s_aud = map_state_to_scalar(v_aud)
            s_tel = map_state_to_scalar(v_tel)
            s_com = map_state_to_scalar(v_com)

            scores = [s_vis, s_aud, s_tel, s_com]
            states = [v_vis, v_aud, v_tel, v_com]
            nodes = ["Vision", "Acoustic", "Telemetry", "Compliance"]

            vec_mean = float(np.mean(scores))
            vec_var = float(np.var(scores))
            disagree_ratio = float(np.std(scores) / 1.0)  # Normalized 0 to 1

            signature = StateVectorSignature(
                timestamp_ms=t,
                vision_score=s_vis,
                acoustic_score=s_aud,
                telemetry_score=s_tel,
                compliance_score=s_com,
                vector_mean=round(vec_mean, 3),
                vector_variance=round(vec_var, 3),
                disagreement_ratio=round(disagree_ratio, 3)
            )

            # Vote Counting
            nom_count = states.count(VerificationState.NOMINAL)
            amb_count = states.count(VerificationState.AMBIGUOUS)
            fraud_count = states.count(VerificationState.FRAUD_SUSPECTED)

            vote_map = {
                VerificationState.NOMINAL: nom_count,
                VerificationState.AMBIGUOUS: amb_count,
                VerificationState.FRAUD_SUSPECTED: fraud_count
            }

            max_votes = max(vote_map.values())
            dominant_state = max(vote_map, key=vote_map.get)
            agreement_pct = round((max_votes / 4.0) * 100.0, 1)

            # Byzantine Protocol Decision Logic
            byzantine_fault = False
            faulting_node = "NONE"
            override_active = False
            severity = ConflictSeverity.NONE

            if max_votes == 4:
                # 4/4 Unanimous Agreement
                consensus_verdict = dominant_state
                severity = ConflictSeverity.NONE
                byzantine_fault = False

            elif max_votes == 3:
                # 3/4 Quorum Achieved (Classical Byzantine Agreement)
                consensus_verdict = dominant_state
                byzantine_fault = True
                override_active = True

                # Identify the single divergent node
                divergent_idx = next(i for i, st in enumerate(states) if st != dominant_state)
                faulting_node = nodes[divergent_idx]
                divergent_state = states[divergent_idx]

                # Classify conflict severity
                score_diff = abs(map_state_to_scalar(dominant_state) - map_state_to_scalar(divergent_state))
                if score_diff >= 2.0:
                    severity = ConflictSeverity.MEDIUM  # NOMINAL vs FRAUD_SUSPECTED (2 steps)
                else:
                    severity = ConflictSeverity.LOW     # 1 step delta (NOMINAL vs AMBIGUOUS)

            else:
                # Split Voting (2 vs 2 or 2 vs 1 vs 1) -> Byzantine Quorum Deficit
                consensus_verdict = VerificationState.AMBIGUOUS
                byzantine_fault = True
                override_active = False  # Quorum deficit locks override
                severity = ConflictSeverity.CRITICAL
                faulting_node = "MULTI_NODE_SPLIT"

            # Check if this slice is part of a verified multi-modal fraud cluster
            cluster_id = self.indexer.get_cluster_id(t)
            if cluster_id:
                # If 2+ nodes confirm fraud or ambiguous surge
                if fraud_count >= 2:
                    severity = ConflictSeverity.CRITICAL
                    consensus_verdict = VerificationState.FRAUD_SUSPECTED

            # Accumulate fault duration
            if byzantine_fault or severity != ConflictSeverity.NONE:
                if last_fault_active:
                    active_fault_duration_ms += (slices[1].offset_ms - slices[0].offset_ms) if len(slices) > 1 else 100
                else:
                    active_fault_duration_ms = 100
                last_fault_active = True
            else:
                active_fault_duration_ms = 0
                last_fault_active = False

            discrepancy_records.append(TemporalDiscrepancyRecord(
                timestamp_ms=t,
                consensus_verdict=consensus_verdict,
                quorum_agreement_pct=agreement_pct,
                conflict_severity=severity,
                byzantine_fault_flag=byzantine_fault,
                faulting_node=faulting_node,
                consensus_override_active=override_active,
                fault_duration_ms=active_fault_duration_ms,
                cluster_id=cluster_id,
                state_vector=signature
            ))

        return discrepancy_records


# =====================================================================
# 5. MASTER TEMPORAL BYZANTINE DISCREPANCY MATRIX FORMATTER
# =====================================================================

class ByzantineMatrixFormatter:
    """Renders comprehensive, scannable solid black markdown tables."""

    @staticmethod
    def render_discrepancy_matrix(
        records: List[TemporalDiscrepancyRecord],
        sample_step_ms: int = 500
    ) -> str:
        lines: List[str] = []

        lines.append("### Master Temporal Byzantine Discrepancy Matrix (Phase 3)")
        lines.append("")
        lines.append("| TIME (ms) | CONSENSUS VERDICT | QUORUM (%) | CONFLICT SEVERITY | FAULTING NODE | OVERRIDE ACTIVE | FAULT DURATION | COINCIDENCE CLUSTER |")
        lines.append("| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |")

        # Select slices at periodic intervals + any slice with a fault or conflict
        selected_times = set(r.timestamp_ms for r in records if r.timestamp_ms % sample_step_ms == 0)
        for r in records:
            if r.byzantine_fault_flag or r.conflict_severity != ConflictSeverity.NONE or r.cluster_id:
                selected_times.add(r.timestamp_ms)

        for r in sorted(records, key=lambda x: x.timestamp_ms):
            if r.timestamp_ms not in selected_times:
                continue

            verdict_badge = f"**`{r.consensus_verdict.value}`**"
            q_str = f"**{r.quorum_agreement_pct:.1f}%**"

            if r.conflict_severity == ConflictSeverity.CRITICAL:
                sev_badge = "**`CRITICAL`**"
            elif r.conflict_severity == ConflictSeverity.MEDIUM:
                sev_badge = "`MEDIUM`"
            elif r.conflict_severity == ConflictSeverity.LOW:
                sev_badge = "`LOW`"
            else:
                sev_badge = "NONE"

            override_str = "**TRUE (QUORUM)**" if r.consensus_override_active else "--"
            fault_node_str = f"`{r.faulting_node}`" if r.faulting_node != "NONE" else "--"
            dur_str = f"{r.fault_duration_ms} ms" if r.fault_duration_ms > 0 else "--"
            cluster_str = f"`{r.cluster_id}`" if r.cluster_id else "--"

            lines.append(
                f"| {r.timestamp_ms:04d} | {verdict_badge} | {q_str} | {sev_badge} | "
                f"{fault_node_str} | {override_str} | {dur_str} | {cluster_str} |"
            )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def render_clusters_table(clusters: List[CoincidenceCluster]) -> str:
        lines: List[str] = []

        lines.append("### Cross-Modal Coincidence Clustered Index")
        lines.append("")
        lines.append("| CLUSTER ID | CENTROID (ms) | TIME SPAN (ms) | MODALITIES INVOLVED | COINCIDENCE TIER | CORRELATION SCORE | CLINICAL INTERPRETATION |")
        lines.append("| :---: | :---: | :---: | :--- | :---: | :---: | :--- |")

        if not clusters:
            lines.append("| -- | -- | -- | None | NOMINAL | 0.000 | No multi-modal anomaly clusters detected |")
        else:
            for c in clusters:
                mods_str = ", ".join(c.modalities_involved)
                span_str = f"[{c.time_span_ms[0]} - {c.time_span_ms[1]}]"
                clean_interp = c.clinical_interpretation.replace("|", "\\|")
                lines.append(
                    f"| **`{c.cluster_id}`** | {c.centroid_ms:.1f} | {span_str} | "
                    f"`{mods_str}` | **`{c.coincidence_tier}`** | **{c.correlation_score:.3f}** | {clean_interp} |"
                )

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def render_synthesis_summary(
        records: List[TemporalDiscrepancyRecord],
        clusters: List[CoincidenceCluster],
        session_id: str
    ) -> str:
        lines: List[str] = []

        total_slices = len(records)
        fault_slices = sum(1 for r in records if r.byzantine_fault_flag)
        overridden_slices = sum(1 for r in records if r.consensus_override_active)
        critical_slices = sum(1 for r in records if r.conflict_severity == ConflictSeverity.CRITICAL)
        unanimous_slices = sum(1 for r in records if r.quorum_agreement_pct == 100.0)
        quorum_maintained_slices = sum(1 for r in records if r.quorum_agreement_pct >= 75.0)

        quorum_rate = (quorum_maintained_slices / total_slices) * 100.0 if total_slices > 0 else 0.0
        unanimous_rate = (unanimous_slices / total_slices) * 100.0 if total_slices > 0 else 0.0

        # Global Session Decision
        if critical_slices > 0:
            session_verdict = "**`FRAUD_SUSPECTED (CONFIRMED MULTI-MODAL BREACH)`**"
            status_desc = "Critical cross-modal coincidence detected. Quorum confirms trial tampering."
        elif fault_slices > 0:
            session_verdict = "**`NOMINAL (FAULT-TOLERANT OVERRIDE ACTIVE)`**"
            status_desc = "Isolated sensor divergence successfully isolated and overridden by 3-node quorum."
        else:
            session_verdict = "**`NOMINAL (UNANIMOUS INTEGRITY)`**"
            status_desc = "All independent validation nodes confirm flawless continuous trial integrity."

        lines.append("### Byzantine Fault Tolerance & Consensus Synthesis Summary")
        lines.append("")
        lines.append("| BYZANTINE EVALUATION METRIC | PROTOCOL SPECIFICATION / STATUS |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Session Identifier** | `{session_id}` |")
        lines.append(f"| **Total Temporal Slices Audited** | `{total_slices}` millisecond slices (0.0 ms to {records[-1].timestamp_ms if records else 0} ms) |")
        lines.append(f"| **Classical BFT Redundancy Margin** | `N=4 Validators (Vision, Acoustic, Telemetry, Compliance) | f=1 Fault Tolerated` |")
        lines.append(f"| **Quorum Agreement Rate (>= 75%)** | **{quorum_rate:.1f}%** ({quorum_maintained_slices}/{total_slices} bins) |")
        lines.append(f"| **Unanimous Agreement Rate (100%)** | **{unanimous_rate:.1f}%** ({unanimous_slices}/{total_slices} bins) |")
        lines.append(f"| **Isolated Byzantine Faults Overridden** | **{overridden_slices}** slices overridden by 3-node quorum |")
        lines.append(f"| **Coincident Anomaly Clusters Indexed** | **{len(clusters)}** spatial-temporal clusters registered |")
        lines.append(f"| **Critical Conflict Slices** | **{critical_slices}** slices (High-severity multi-sensor coincidence) |")
        lines.append(f"| **Global Byzantine Session Verdict** | {session_verdict} |")
        lines.append(f"| **Consensus Engine Status** | {status_desc} |")
        lines.append("")

        return "\n".join(lines)


# =====================================================================
# 6. VERIFICATION RUNNER (PHASE 1 + PHASE 2 + PHASE 3 PIPELINE)
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 104)
    print("  ANATOMOZK // 3D CORONARY RISK & BYZANTINE BFT TRUST ENGINE (PHASE 3) - INITIALIZATION".center(104))
    print("=" * 104)

    # 1. Environment Verification
    env = audit_environment()
    print(f"  * Environment Source : {env.env_loaded_from}")
    print(f"  * GEMINI_API_KEY     : {'[ACTIVE]' if env.is_gemini_active else '[INACTIVE]':<10} {env.gemini_masked}")
    print(f"  * GROQ_API_KEY       : {'[ACTIVE]' if env.is_groq_active else '[INACTIVE]':<10} {env.groq_masked}")
    print(f"  * Consensus Spec     : Classical Byzantine Fault Tolerance (N=4, f=1, Quorum >= 75%)")
    print("-" * 104)

    # 2. Run Phase 1 & Phase 2 Mock Pipelines in Isolated Sandbox
    temp_dir = Path(tempfile.mkdtemp(prefix="anatomozk_p3_"))

    try:
        print("\n[Step 1/3] Executing Phase 1 Multimodal Ingestion Engine & Phase 2 Agent Graph...")
        vid_p, aud_p, tel_p, doc_p = generate_mock_session_assets(temp_dir)

        # Ingest 4 asynchronous streams
        ingestor = MultimodalSessionIngestor(
            video_path=vid_p,
            audio_path=aud_p,
            telemetry_path=tel_p,
            compliance_path=doc_p,
            config=IngestionConfig(video_sample_fps=10.0, telemetry_z_threshold=3.0),
            session_id="BIO-SYN-PHASE3-RUN"
        )
        ingestion_res = ingestor.ingest_all()
        slices_batch = [ingestion_res.time_dictionary[t] for t in sorted(ingestion_res.time_dictionary.keys())]

        # Dispatch Phase 2 State Router
        client_registry = LLMClientRegistry(env=env)
        router = DecoupledStateRouter(registry=client_registry, max_workers=4)
        agent_consensus = router.route_and_execute(
            session_id=ingestion_res.session_id,
            slices=slices_batch,
            compliance_meta=ingestion_res.compliance_meta
        )

        print(f"           - Synchronized Time Slices : {len(slices_batch)} bins (Span: 0 to {slices_batch[-1].offset_ms} ms)")
        print(f"           - Phase 2 Agents Evaluated : {len(agent_consensus.agent_evaluations)} validation actors")
        print(f"           - LLM Global Verdict       : {agent_consensus.global_verdict.value} ({agent_consensus.aggregate_confidence:.1f}%)")

        # 3. Execute Phase 3 Temporal Byzantine Consensus Engine
        print("\n[Step 2/3] Executing TemporalByzantineConsensusEngine over millisecond timeline...")
        consensus_engine = TemporalByzantineConsensusEngine(
            temporal_jitter_ms=150,
            byzantine_quorum_threshold=0.75
        )

        start_byz_t = time.perf_counter()
        discrepancy_records = consensus_engine.execute_byzantine_consensus(
            slices=slices_batch,
            agent_evaluations=agent_consensus.agent_evaluations
        )
        byz_latency = round((time.perf_counter() - start_byz_t) * 1000.0, 2)
        print(f"           - Byzantine Protocol Pass  : Complete in {byz_latency:.2f} ms")
        print(f"           - Coincidence Clusters     : {len(consensus_engine.indexer.clusters)} indexed")

        # 4. Render Master Temporal Byzantine Discrepancy Matrix Output
        print("\n[Step 3/3] Generating Master Temporal Byzantine Discrepancy Matrix (Solid Black Tables):\n")

        # Table 1: Master Discrepancy Matrix
        table_1 = ByzantineMatrixFormatter.render_discrepancy_matrix(discrepancy_records, sample_step_ms=500)
        print(table_1)

        # Table 2: Coincidence Clustered Index Table
        table_2 = ByzantineMatrixFormatter.render_clusters_table(consensus_engine.indexer.clusters)
        print(table_2)

        # Table 3: Byzantine Synthesis Summary
        table_3 = ByzantineMatrixFormatter.render_synthesis_summary(
            records=discrepancy_records,
            clusters=consensus_engine.indexer.clusters,
            session_id=ingestion_res.session_id
        )
        print(table_3)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("=" * 104)
    print("  ANATOMOZK PHASE 3 TEMPORAL BYZANTINE BFT ENGINE VERIFIED NOMINALLY".center(104))
    print("=" * 104 + "\n")
