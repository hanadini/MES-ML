package mom.decision;

public record EngineState(
        RiskState state,
        Persistency persistency
) {
    public static EngineState initial() {
        return new EngineState(RiskState.NORMAL, Persistency.empty());
    }
}