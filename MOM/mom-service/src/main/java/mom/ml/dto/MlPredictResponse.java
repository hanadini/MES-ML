package mom.ml.dto;

import org.springframework.boot.autoconfigure.mail.MailProperties;

import java.util.Map;

public record MlPredictResponse(
        String request_id,
        String timestamp,
        String line_id,
        Map<String, Object> prediction,
        Map<String, Object> explainability,
        Map<String, Object> data_quality,
        Map<String, Object> model
) {
    public double probability() {
        Object p = prediction.get("scrap_risk_probability");
        return p instanceof Number ? ((Number) p).doubleValue() : 0.0;
    }

    public String riskLevel() {
        Object r = prediction.get("risk_level");
        return r != null ? r.toString() : "UNKNOWN";
    }

    public double dataQualityScore() {
        Object dq = data_quality.get("quality_score");
        return dq instanceof Number ? ((Number) dq).doubleValue() : 0.0;
    }

}
