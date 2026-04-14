package mom.dto;

import java.util.List;

public record RiskEvaluateResponse(
        String requestId,
        String lineId,
        String appliedState,
        double probability,
        String riskLevel,
        double appliedThreshold,
        boolean fallbackUsed,
        List<String> actionsCreated
) {}
