from app.schemas.prediction import TrumpInput, WarPredictionResponse


class TrumpWarPredictor:
    """
    트럼프 전쟁 확률 Mock 예측 모델.

    실제 ML 모델 자리지만, 오늘은 단순 규칙 기반 로직으로 대체합니다.
    스트레스가 높고, 빅맥을 많이 먹고, 잠을 적게 잘수록 위험도가 올라갑니다.
    """

    VERSION = "trump-predictor-v1.0"

    def predict(self, user_input: TrumpInput) -> WarPredictionResponse:
        risk = self._compute_risk(user_input)
        label = self._pick_label(risk)
        reason = self._make_reason(user_input, label)

        return WarPredictionResponse(
            label=label,
            probability=round(risk, 4),
            reason=reason,
            model_version=self.VERSION,
        )

    def _compute_risk(self, user_input: TrumpInput) -> float:
        stress_score = user_input.trump_stress_index / 100
        big_mac_score = user_input.big_mac_count / 10
        sleep_penalty = max(0.0, (8 - user_input.sleep_hours) / 8)

        risk = 0.5 * stress_score + 0.3 * big_mac_score + 0.2 * sleep_penalty
        return min(max(risk, 0.0), 1.0)

    def _pick_label(self, risk: float) -> str:
        if risk < 0.15:
            return "peaceful"
        if risk < 0.35:
            return "tariff_war"
        if risk < 0.55:
            return "trade_war"
        if risk < 0.75:
            return "diplomatic_crisis"
        if risk < 0.9:
            return "limited_strike"
        return "world_war_3"

    def _make_reason(self, user_input: TrumpInput, label: str) -> str:
        s = user_input.trump_stress_index
        b = user_input.big_mac_count
        h = user_input.sleep_hours

        if label == "peaceful":
            return f"스트레스 {s}, 잠 {h}시간 → 오늘은 골프나 치러 가심 🏌️"
        if label == "tariff_war":
            return f"빅맥 {b}개로 컨디션 회복 → 가벼운 관세 폭탄 예상 💰"
        if label == "trade_war":
            return f"스트레스 {s} + 빅맥 {b}개 → 관세 200% 트윗 임박 📉"
        if label == "diplomatic_crisis":
            return f"잠 {h}시간 → 새벽 3시 트루스소셜 폭주 예상 🤬"
        if label == "limited_strike":
            return f"스트레스 {s}, 수면 {h}시간 → 누군가에게 미사일 한 발 ✈️"
        return f"빅맥 {b}개 + 잠 {h}시간 + 스트레스 {s} → 핵 단추 위험 ☢️"


ml_model = TrumpWarPredictor()
