package mom.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.Map;

public record RiskEvaluateRequest(
        @NotBlank String requestId,
        @NotBlank String timestamp,
        @NotBlank String plant,
        @NotBlank String lineId,
        @NotNull Integer horizonMinutes,
        @NotNull Integer windowMinutes,
        @NotNull Map<String, String> context,
        @NotNull Map<String, Double> features
) {}

