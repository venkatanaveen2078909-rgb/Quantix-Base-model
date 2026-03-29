"""Quantix v3 — Layer 2: Delay Prediction Agent"""
from __future__ import annotations
import logging
from models.schemas import RouteInput, WeatherOutput, TrafficOutput, DelayOutput
from events.event_bus import AgentEventBus
from events.event_types import EventType, EventSeverity, QuantixEvent

logger = logging.getLogger("quantix.DelayAgent")

class DelayPredictionAgent:
    def __init__(self, bus: AgentEventBus):
        self.bus = bus

    async def run(self, ri: RouteInput, weather: WeatherOutput, traffic: TrafficOutput) -> DelayOutput:
        delays, probs, critical = {}, {}, []
        total_delay = 0.0

        for node in ri.nodes:
            w_risk = weather.weather_risk_scores.get(node, 0.1)
            t_risk = sum(v for k,v in traffic.congestion_scores.items() if k.endswith(node)) / 5 or 0.1
            delay = round((w_risk * 45) + (t_risk * 30), 1)
            delays[node] = delay
            probs[node] = round(min(w_risk + t_risk, 1.0), 3)
            total_delay += delay
            if delay > 25:
                critical.append(node)

        return DelayOutput(
            predicted_delays=delays,
            delay_probs=probs,
            critical_segments=critical,
            total_expected_delay_min=round(total_delay, 1),
            on_time_probability=round(max(0.0, 1.0 - total_delay/500), 3),
            model_summary=f"Expected delay: {total_delay:.1f}m. On-time prob: {1.0 - total_delay/500:.0%}. Critical: {len(critical)}."
        )
