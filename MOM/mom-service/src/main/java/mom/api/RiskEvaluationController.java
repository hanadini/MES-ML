package mom.api;
import jakarta.validation.Valid;
import mom.dto.RiskEvaluateRequest;
import mom.dto.RiskEvaluateResponse;
import mom.service.RiskOrchestratorService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/risk")
public class RiskEvaluationController {

    private final RiskOrchestratorService orchestratorService;

    public RiskEvaluationController(RiskOrchestratorService orchestratorService) {
        this.orchestratorService = orchestratorService;
    }

    @PostMapping("/evaluate")
    public RiskEvaluateResponse evaluate(@Valid @RequestBody RiskEvaluateRequest request) {
        return orchestratorService.evaluateRisk(request);
    }
}
