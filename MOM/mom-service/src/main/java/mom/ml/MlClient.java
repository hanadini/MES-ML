package mom.ml;

import mom.ml.dto.MlPredictRequest;
import mom.ml.dto.MlPredictResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Map;

@Component
public class MlClient {
    private final WebClient webClient;
    private final Duration timeout;

    public MlClient(WebClient webClient, @Value("${ml.timeout-ms:800}") long timeoutMs) {
        this.webClient = webClient;
        this.timeout = Duration.ofMillis(timeoutMs); // default timeout, can be made configurable
    }

    public MlPredictResponse predictScrapRisk(MlPredictRequest request){
        return webClient.post()
                .uri("/v1/predict/scrap-risk")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(MlPredictResponse.class)
                .timeout(timeout) // apply timeout to the request
                .onErrorReturn(new MlPredictResponse(
                        request.request_id(),
                        request.timestamp(),
                        request.line_id(),
                        Map.of("scrap_risk_probability", 0.0, "risk_level", "UNKNOWN"),
                        Map.of(),
                        Map.of("quality_score", 0.0),
                        Map.of()
                )) // return default response on error
                .block();
    }

}
