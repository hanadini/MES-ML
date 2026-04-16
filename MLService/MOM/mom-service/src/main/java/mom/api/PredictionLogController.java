package mom.api;

import mom.domain.PredictionLog;
import mom.service.PredictionLogQueryService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/v1/predictions")
public class PredictionLogController {

    private final PredictionLogQueryService predictionLogQueryService;

    public PredictionLogController(PredictionLogQueryService predictionLogQueryService) {
        this.predictionLogQueryService = predictionLogQueryService;
    }

    @GetMapping
    public List<PredictionLog> getPredictionLogs(
            @RequestParam(required = false) String lineId,
            @RequestParam(required = false) String orderId,
            @RequestParam(required = false) String requestId
    ) {
        return predictionLogQueryService.getLogs(lineId, orderId, requestId);
    }
}