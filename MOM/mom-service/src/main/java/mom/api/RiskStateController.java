package mom.api;

import mom.domain.RiskStateEntity;
import mom.service.RiskStateQueryService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/state")
public class RiskStateController {

    private final RiskStateQueryService riskStateQueryService;

    public RiskStateController(RiskStateQueryService riskStateQueryService) {
        this.riskStateQueryService = riskStateQueryService;
    }

    @GetMapping("/{lineId}/{orderId}")
    public RiskStateEntity getState(
            @PathVariable String lineId,
            @PathVariable String orderId
    ) {
        return riskStateQueryService.getState(lineId, orderId);
    }
}