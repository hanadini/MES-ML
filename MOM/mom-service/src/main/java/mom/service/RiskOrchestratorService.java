package mom.service;

import mom.decision.ActionType;
import mom.decision.DecisionContext;
import mom.decision.DecisionEngine;
import mom.decision.DecisionResult;
import mom.decision.EngineState;
import mom.decision.Persistency;
import mom.decision.RiskState;
import mom.domain.MomEvent;
import mom.domain.MomTask;
import mom.domain.PredictionLog;
import mom.domain.RiskStateEntity;
import mom.dto.RiskEvaluateRequest;
import mom.dto.RiskEvaluateResponse;
import mom.ml.MlClient;
import mom.ml.dto.MlPredictRequest;
import mom.ml.dto.MlPredictResponse;
import mom.repo.MomEventRepository;
import mom.repo.MomTaskRepository;
import mom.repo.PredictionLogRepository;
import mom.repo.RiskStateRepository;
import mom.util.TimeUtil;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class RiskOrchestratorService {

    private final MlClient mlClient;
    private final DecisionEngine decisionEngine;
    private final RiskStateRepository riskStateRepository;
    private final MomEventRepository momEventRepository;
    private final MomTaskRepository momTaskRepository;
    private final PredictionLogRepository predictionLogRepository;

    public RiskOrchestratorService(MlClient mlClient,
                                   DecisionEngine decisionEngine,
                                   RiskStateRepository riskStateRepository,
                                   MomEventRepository momEventRepository,
                                   MomTaskRepository momTaskRepository,
                                   PredictionLogRepository predictionLogRepository) {
        this.mlClient = mlClient;
        this.decisionEngine = decisionEngine;
        this.riskStateRepository = riskStateRepository;
        this.momEventRepository = momEventRepository;
        this.momTaskRepository = momTaskRepository;
        this.predictionLogRepository = predictionLogRepository;
    }

    @Transactional
    public RiskEvaluateResponse evaluateRisk(RiskEvaluateRequest request) {

        MlPredictRequest mlReq = new MlPredictRequest(
                request.requestId(),
                request.timestamp(),
                request.plant(),
                request.lineId(),
                request.horizonMinutes(),
                request.windowMinutes(),
                request.context(),
                request.features()
        );

        MlPredictResponse mlRes = mlClient.predictScrapRisk(mlReq);

        String orderId = request.context().getOrDefault("orderId", "UNKNOWN");
        String shift = request.context().getOrDefault("shift", "*");
        String recipe = request.context().getOrDefault("recipe", "*");

        DecisionContext decisionContext = new DecisionContext(
                request.lineId(),
                orderId,
                shift,
                recipe
        );

        RiskStateEntity savedState = riskStateRepository
                .findByLineIdAndOrderId(request.lineId(), orderId)
                .orElse(new RiskStateEntity(
                        request.lineId(),
                        orderId,
                        "NORMAL",
                        0,
                        0
                ));

        EngineState currentState = new EngineState(
                RiskState.valueOf(savedState.getCurrentState()),
                new Persistency(savedState.getWarnCount(), savedState.getHighCount())
        );

        DecisionResult decisionResult = decisionEngine.evaluate(
                decisionContext,
                currentState,
                mlRes.probability(),
                mlRes.dataQualityScore()
        );

        savedState.setCurrentState(decisionResult.newState().name());
        savedState.setWarnCount(decisionResult.newPersistency().warnCount());
        savedState.setHighCount(decisionResult.newPersistency().highCount());
        riskStateRepository.save(savedState);

        PredictionLog predictionLog = new PredictionLog(
                request.requestId(),
                request.lineId(),
                orderId,
                mlRes.probability(),
                mlRes.riskLevel(),
                decisionResult.newState().name(),
                decisionResult.appliedHighThreshold(),
                mlRes.dataQualityScore(),
                decisionResult.fallbackUsed(),
                extractModelVersion(mlRes),
                TimeUtil.nowUtc()
        );
        predictionLogRepository.save(predictionLog);

        boolean createEvent = decisionResult.actions().contains(ActionType.CREATE_EVENT);
        boolean createQualityTask = decisionResult.actions().contains(ActionType.CREATE_QUALITY_TASK);
        boolean requestSupervisorApproval = decisionResult.actions().contains(ActionType.REQUEST_SUPERVISOR_APPROVAL);

        if (createEvent) {
            MomEvent event = new MomEvent(
                    "Predictive_Quality_Risk",
                    request.lineId(),
                    orderId,
                    mlRes.riskLevel(),
                    mlRes.probability(),
                    TimeUtil.nowUtc()
            );
            momEventRepository.save(event);
        }

        if (createQualityTask || requestSupervisorApproval) {
            String taskType = requestSupervisorApproval ? "SUPERVISOR_APPROVAL" : "QUALITY_CHECK";

            MomTask task = new MomTask(
                    taskType,
                    request.lineId(),
                    orderId,
                    "OPEN",
                    "UNASSIGNED",
                    TimeUtil.nowUtc()
            );
            momTaskRepository.save(task);
        }

        List<String> actions = decisionResult.actions()
                .stream()
                .map(Enum::name)
                .toList();

        return new RiskEvaluateResponse(
                request.requestId(),
                request.lineId(),
                decisionResult.newState().name(),
                mlRes.probability(),
                mlRes.riskLevel(),
                decisionResult.appliedHighThreshold(),
                decisionResult.fallbackUsed(),
                actions
        );
    }

    private String extractModelVersion(MlPredictResponse mlRes) {
        Object version = mlRes.model().get("model_version");
        return version != null ? version.toString() : "UNKNOWN";
    }
}