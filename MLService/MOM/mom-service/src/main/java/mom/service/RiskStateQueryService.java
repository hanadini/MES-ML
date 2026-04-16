package mom.service;

import mom.domain.RiskStateEntity;
import mom.repo.RiskStateRepository;
import org.springframework.stereotype.Service;

@Service
public class RiskStateQueryService {

    private final RiskStateRepository riskStateRepository;

    public RiskStateQueryService(RiskStateRepository riskStateRepository) {
        this.riskStateRepository = riskStateRepository;
    }

    public RiskStateEntity getState(String lineId, String orderId) {
        return riskStateRepository.findByLineIdAndOrderId(lineId, orderId)
                .orElseThrow(() -> new RuntimeException(
                        "Risk state not found for lineId=" + lineId + ", orderId=" + orderId
                ));
    }
}
