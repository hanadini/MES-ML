package mom.decision;

import java.util.ArrayList;
import java.util.List;

public class DecisionEngine {
    private final ThresholdPolicy thresholdPolicy;

    // Tune these in config later:
    private final double dataQualityMin = 0.85;
    private final int warnPersistN = 2;   // need 2 consecutive warns
    private final int highPersistN = 3;   // need 3 consecutive highs
    private final int coolDownToNormal = 3; // consecutive below-warn to reset

    public DecisionEngine(ThresholdPolicy thresholdPolicy) {
        this.thresholdPolicy = thresholdPolicy;
    }

    public DecisionResult evaluate(
            DecisionContext ctx,
            EngineState current,
            double probability,
            double dataQualityScore
    ) {
        Thresholds thresholds = thresholdPolicy.thresholdsFor(ctx.lineId(), ctx.shift(), ctx.recipe());
        double warnT = thresholds.warnThreshold();
        double highT = thresholds.highThreshold();

        List<ActionType> actions = new ArrayList<>();

        // 0) Safety gate — fallback
        if (dataQualityScore < dataQualityMin) {
            actions.add(ActionType.FALLBACK_RULE_ALERT);
            actions.add(ActionType.LOG_ONLY);

            return new DecisionResult(
                    RiskState.FALLBACK_MODE,
                    Persistency.empty(),
                    warnT,
                    highT,
                    true,
                    actions
            );
        }

        // classify this cycle
        boolean isHigh = probability >= highT;
        boolean isWarn = probability >= warnT && probability < highT;
        boolean isBelowWarn = probability < warnT;

        Persistency p = current.persistency();

        int newWarnCount = isWarn ? p.warnCount() + 1 : (isBelowWarn ? 0 : p.warnCount());
        int newHighCount = isHigh ? p.highCount() + 1 : 0;

        Persistency nextPersist = new Persistency(newWarnCount, newHighCount);

        RiskState currentState = current.state();
        RiskState nextState = currentState;

        // always log prediction
        actions.add(ActionType.LOG_ONLY);

        // --- State transitions ---
        switch (currentState) {

            case NORMAL -> {
                if (isWarn && newWarnCount >= warnPersistN) {
                    nextState = RiskState.MONITORING;
                    actions.add(ActionType.CREATE_EVENT);
                }
                if (isHigh) {
                    // high immediately bumps to WARNING (but no escalation until persistency)
                    nextState = RiskState.WARNING;
                    actions.add(ActionType.CREATE_EVENT);
                }
            }

            case MONITORING -> {
                if (isHigh) {
                    nextState = RiskState.WARNING;
                    actions.add(ActionType.CREATE_EVENT);
                } else if (isBelowWarn && coolDownHit(p.warnCount(), newWarnCount)) {
                    nextState = RiskState.NORMAL;
                } else {
                    // keep monitoring, no spam
                }
            }

            case WARNING -> {
                actions.add(ActionType.CREATE_QUALITY_TASK); // non-blocking task in warning
                if (isHigh && newHighCount >= 2) {
                    nextState = RiskState.ESCALATED;
                    actions.add(ActionType.NOTIFY_SHIFT_LEAD);
                } else if (isBelowWarn && coolDownHit(p.warnCount(), newWarnCount)) {
                    nextState = RiskState.RESOLVED;
                }
            }

            case ESCALATED -> {
                actions.add(ActionType.NOTIFY_SHIFT_LEAD);
                actions.add(ActionType.CREATE_QUALITY_TASK);

                if (isHigh && newHighCount >= highPersistN) {
                    nextState = RiskState.INTERVENTION_PENDING;
                    actions.add(ActionType.REQUEST_SUPERVISOR_APPROVAL);
                } else if (isBelowWarn && coolDownHit(p.warnCount(), newWarnCount)) {
                    nextState = RiskState.RESOLVED;
                }
            }

            case INTERVENTION_PENDING -> {
                // In real system, this state waits for supervisor decision.
                // Here, we keep it until external approval updates the state.
                actions.add(ActionType.REQUEST_SUPERVISOR_APPROVAL);
            }

            case INTERVENTION_ACTIVE -> {
                // Similar: transitions depend on measured improvement and supervisor logic.
                // For MVP: if below warn for a while => resolved.
                if (isBelowWarn && coolDownHit(p.warnCount(), newWarnCount)) {
                    nextState = RiskState.RESOLVED;
                } else {
                    actions.add(ActionType.ACTIVATE_INTERVENTION);
                }
            }

            case RESOLVED -> {
                // After resolved, drop to normal on next stable cycle
                if (isBelowWarn) {
                    nextState = RiskState.NORMAL;
                    nextPersist = Persistency.empty();
                } else if (isWarn) {
                    nextState = RiskState.MONITORING;
                } else if (isHigh) {
                    nextState = RiskState.WARNING;
                }
            }

            case FALLBACK_MODE -> {
                // If data quality recovered, re-enter NORMAL
                nextState = RiskState.NORMAL;
                nextPersist = Persistency.empty();
            }
        }

        return new DecisionResult(
                nextState,
                nextPersist,
                warnT,
                highT,
                false,
                actions
        );
    }

    private boolean coolDownHit(int previousWarnCount, int newWarnCount) {
        // For MVP: if below warn, warn count resets to 0; we use that as cooldown.
        // You can replace with belowWarnPersist counter later.
        return newWarnCount == 0;
    }
}
