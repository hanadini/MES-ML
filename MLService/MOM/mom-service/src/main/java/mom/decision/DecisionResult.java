package mom.decision;

import java.util.List;

public record DecisionResult(
        RiskState newState,
        Persistency newPersistency,
        double appliedWarnThreshold,
        double appliedHighThreshold,
        boolean fallbackUsed,
        List<ActionType> actions
) {}