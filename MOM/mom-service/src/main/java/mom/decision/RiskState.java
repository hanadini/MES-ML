package mom.decision;

public enum RiskState {
    NORMAL,
    MONITORING,
    WARNING,
    ESCALATED,
    INTERVENTION_PENDING,
    INTERVENTION_ACTIVE,
    RESOLVED,
    FALLBACK_MODE
}
