package mom.ml.dto;

import java.util.Map;

public record MlPredictRequest(
        String request_id,
        String timestamp,
        String plant,
        String line_id,
        int horizon_minutes,
        int window_minutes,
        Map<String, String> context,
        Map<String, Double> features
) {}
