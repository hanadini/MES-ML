package mom.decision;

public record DecisionContext(
        String lineId,
        String orderId,
        String shift,
        String recipe
) {}
