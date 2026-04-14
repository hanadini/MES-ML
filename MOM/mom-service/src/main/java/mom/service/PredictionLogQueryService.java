package mom.service;

import mom.domain.PredictionLog;
import mom.repo.PredictionLogRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class PredictionLogQueryService {

    private final PredictionLogRepository predictionLogRepository;

    public PredictionLogQueryService(PredictionLogRepository predictionLogRepository) {
        this.predictionLogRepository = predictionLogRepository;
    }

    public List<PredictionLog> getLogs(String lineId, String orderId, String requestId) {
        if (requestId != null && !requestId.isBlank()) {
            return predictionLogRepository.findByRequestId(requestId);
        }
        if (lineId != null && !lineId.isBlank()) {
            return predictionLogRepository.findByLineId(lineId);
        }
        if (orderId != null && !orderId.isBlank()) {
            return predictionLogRepository.findByOrderId(orderId);
        }
        return predictionLogRepository.findAll();
    }
}
